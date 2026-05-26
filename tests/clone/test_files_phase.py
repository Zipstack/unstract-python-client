"""Tests for ``FilesPhase``.

Coverage:
- happy path: PDF + text/csv files uploaded with base64 + utf-8 decoding.
- target-side idempotency: filename already present → skip, no upload.
- oversize file → ``oversize_files`` entry, sibling files continue.
- unsupported mime (Excel placeholder) → ``unsupported_files`` entry.
- skip strategy → no uploads, source filenames listed in ``skipped_files``.
- dry-run → no uploads even for missing files.
- transient 503 → retried, eventual success.
- no custom_tool remap → no-op.
- listing failure on source aborts only that tool, others continue.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from unstract.clone.context import (
    CloneContext,
    CloneOptions,
    OrgEndpoint,
    RemapTable,
)
from unstract.clone.exceptions import PlatformAPIError
from unstract.clone.phases.files import FilesPhase
from unstract.clone.report import CloneReport

SRC_ENDPOINT = OrgEndpoint(
    base_url="http://src", organization_id="src-org", platform_key="src-key"
)
TGT_ENDPOINT = OrgEndpoint(
    base_url="http://tgt", organization_id="tgt-org", platform_key="tgt-key"
)


class FakeClient:
    def __init__(
        self,
        *,
        endpoint: OrgEndpoint,
        documents: dict[str, list[dict]] | None = None,
        file_payloads: dict[tuple[str, str], dict] | None = None,
        tools: list[dict] | None = None,
    ):
        self.endpoint = endpoint
        # tool_id -> list of {document_name, document_id, tool}
        self._documents: dict[str, list[dict]] = {
            k: list(v) for k, v in (documents or {}).items()
        }
        # (tool_id, file_name) -> {"data": ..., "mime_type": ...}
        self._file_payloads: dict[tuple[str, str], dict] = dict(file_payloads or {})
        self._tools = list(tools or [])
        self.uploaded: list[dict[str, Any]] = []
        self.list_calls: list[str] = []
        self.download_calls: list[tuple[str, str]] = []
        # Configurable fault injection.
        self.download_errors: dict[tuple[str, str], list[Exception]] = {}
        self.upload_errors: dict[tuple[str, str], list[Exception]] = {}
        self.list_errors: dict[str, Exception] = {}
        self._next_id = 1

    def list_prompt_documents(self, tool_id: str) -> list[dict]:
        self.list_calls.append(tool_id)
        if tool_id in self.list_errors:
            raise self.list_errors[tool_id]
        return [dict(d) for d in self._documents.get(tool_id, [])]

    def download_prompt_file(self, tool_id: str, document_id: str) -> dict:
        # Tests key payloads + error queues by (tool_id, file_name) for
        # readability; resolve the filename from the documents list.
        file_name = next(
            (
                d["document_name"]
                for d in self._documents.get(tool_id, [])
                if d.get("document_id") == document_id
            ),
            document_id,
        )
        self.download_calls.append((tool_id, file_name))
        queue = self.download_errors.get((tool_id, file_name))
        if queue:
            raise queue.pop(0)
        return dict(self._file_payloads[(tool_id, file_name)])

    def upload_prompt_file(
        self, tool_id: str, file_name: str, data: bytes, mime_type: str
    ) -> dict:
        queue = self.upload_errors.get((tool_id, file_name))
        if queue:
            raise queue.pop(0)
        doc_id = f"doc-{self._next_id:04d}"
        self._next_id += 1
        self.uploaded.append(
            {
                "tool_id": tool_id,
                "file_name": file_name,
                "data": data,
                "mime_type": mime_type,
            }
        )
        self._documents.setdefault(tool_id, []).append(
            {"document_id": doc_id, "document_name": file_name, "tool": tool_id}
        )
        return {"document_id": doc_id}

    def list_custom_tools(self) -> list[dict]:
        return list(self._tools)

    def get_custom_tool(self, tool_id: str) -> dict:
        return dict(next((t for t in self._tools if t.get("tool_id") == tool_id), {}))

    def update_custom_tool(self, tool_id: str, body: dict) -> dict:
        for t in self._tools:
            if t.get("tool_id") == tool_id:
                t.update(body)
                return dict(t)
        return {}


def _ctx(
    src: FakeClient, tgt: FakeClient, *, remap: RemapTable | None = None, **opts
) -> CloneContext:
    remap = remap or RemapTable()
    return CloneContext(
        source=src,
        target=tgt,
        options=CloneOptions(**opts),
        remap=remap,
    )


def _doc(name: str) -> dict:
    return {"document_id": f"src-{name}", "document_name": name, "tool": "ignored"}


def _pdf_payload(raw: bytes) -> dict:
    return {"data": base64.b64encode(raw).decode(), "mime_type": "application/pdf"}


def _text_payload(text: str, mime: str = "text/plain") -> dict:
    return {"data": text, "mime_type": mime}


def test_happy_path_uploads_pdf_and_text():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("invoice.pdf"), _doc("notes.txt")]},
        file_payloads={
            ("src-1", "invoice.pdf"): _pdf_payload(b"%PDF-FAKE"),
            ("src-1", "notes.txt"): _text_payload("hello world"),
        },
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.created == 2
    assert result.failed == 0
    assert {u["file_name"] for u in tgt.uploaded} == {"invoice.pdf", "notes.txt"}
    pdf_upload = next(u for u in tgt.uploaded if u["file_name"] == "invoice.pdf")
    assert pdf_upload["data"] == b"%PDF-FAKE"
    assert pdf_upload["mime_type"] == "application/pdf"
    txt_upload = next(u for u in tgt.uploaded if u["file_name"] == "notes.txt")
    assert txt_upload["data"] == b"hello world"
    assert len(report.uploaded_files) == 2
    assert all(u["tool_name"] == "demo" for u in report.uploaded_files)


def test_target_filename_present_is_skipped_no_download():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("invoice.pdf")]},
        file_payloads={("src-1", "invoice.pdf"): _pdf_payload(b"BYTES")},
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT,
        documents={"tgt-1": [_doc("invoice.pdf")]},
        tools=[{"tool_id": "tgt-1", "tool_name": "demo"}],
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.skipped == 1
    assert result.created == 0
    assert tgt.uploaded == []
    assert src.download_calls == []  # pre-check guards the download


def test_oversize_file_is_recorded_and_siblings_continue():
    big = b"X" * 50
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("big.pdf"), _doc("small.txt")]},
        file_payloads={
            ("src-1", "big.pdf"): _pdf_payload(big),
            ("src-1", "small.txt"): _text_payload("ok"),
        },
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap, max_file_size=10)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.created == 1
    # Oversize must bump skipped so the operator sees it surfaced in the
    # phase counters, not only in the report's list.
    assert result.skipped == 1
    assert result.failed == 0
    assert {u["file_name"] for u in tgt.uploaded} == {"small.txt"}
    assert len(report.oversize_files) == 1
    over = report.oversize_files[0]
    assert over["file_name"] == "big.pdf"
    assert over["size_bytes"] == 50
    assert over["cap_bytes"] == 10


def test_unsupported_mime_is_recorded_not_uploaded():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("sheet.xlsx")]},
        file_payloads={
            ("src-1", "sheet.xlsx"): {
                "data": "Preview not available for Excel files. ...",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }
        },
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.created == 0
    # Unsupported mimes must bump skipped so the run doesn't report green
    # while leaving files unmoved.
    assert result.skipped == 1
    assert result.failed == 0
    assert tgt.uploaded == []
    assert len(report.unsupported_files) == 1
    entry = report.unsupported_files[0]
    assert entry["file_name"] == "sheet.xlsx"
    assert entry["mime_type"].startswith("application/vnd.openxmlformats")


def test_malformed_source_dm_row_bumps_skipped_with_error():
    # Renamed-field or partial-serializer response: row lacks
    # document_name/document_id. Must surface, not silently disappear.
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [{"tool": "src-1"}, _doc("ok.pdf")]},
        file_payloads={("src-1", "ok.pdf"): _pdf_payload(b"BYTES")},
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.created == 1  # the well-formed sibling still uploads.
    assert result.skipped == 1  # the malformed row.
    assert any("malformed source DM row" in e for e in result.errors)


def test_skip_strategy_emits_skipped_files_no_traffic():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("a.pdf"), _doc("b.pdf")]},
    )
    tgt = FakeClient(endpoint=TGT_ENDPOINT)
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap, file_strategy="skip")
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.skipped == 2
    assert tgt.uploaded == []
    assert src.download_calls == []
    names = {row["file_name"] for row in report.skipped_files}
    assert names == {"a.pdf", "b.pdf"}
    assert all(row["source_org_slug"] == "src-org" for row in report.skipped_files)
    assert all(row["source_tool_id"] == "src-1" for row in report.skipped_files)


def test_dry_run_makes_no_writes_even_for_missing_files():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("a.pdf")]},
        file_payloads={("src-1", "a.pdf"): _pdf_payload(b"X")},
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap, dry_run=True)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.skipped == 1
    assert result.created == 0
    assert tgt.uploaded == []
    assert src.download_calls == []


def test_transient_503_is_retried_then_succeeds(monkeypatch):
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("a.pdf")]},
        file_payloads={("src-1", "a.pdf"): _pdf_payload(b"OK")},
    )
    src.download_errors[("src-1", "a.pdf")] = [
        PlatformAPIError("flaky", status_code=503, body="")
    ]
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)
    # Strip the backoff sleep so the test stays fast.
    monkeypatch.setattr("unstract.clone.phases.files.time.sleep", lambda *_: None)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.created == 1
    assert tgt.uploaded[0]["data"] == b"OK"


def test_no_custom_tool_remap_is_noop():
    src = FakeClient(endpoint=SRC_ENDPOINT)
    tgt = FakeClient(endpoint=TGT_ENDPOINT)
    ctx = _ctx(src, tgt)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.created == 0
    assert result.skipped == 0
    assert src.list_calls == []


def test_source_list_failure_isolates_to_that_tool():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-2": [_doc("ok.pdf")]},
        file_payloads={("src-2", "ok.pdf"): _pdf_payload(b"OK")},
    )
    src.list_errors["src-1"] = RuntimeError("source down for this tool")
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT,
        tools=[
            {"tool_id": "tgt-1", "tool_name": "broken"},
            {"tool_id": "tgt-2", "tool_name": "healthy"},
        ],
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    remap.record("custom_tool", "src-2", "tgt-2")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.failed == 1
    assert result.created == 1
    assert {u["file_name"] for u in tgt.uploaded} == {"ok.pdf"}


def test_upload_failure_records_failed_files_entry():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("a.pdf")]},
        file_payloads={("src-1", "a.pdf"): _pdf_payload(b"X")},
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    tgt.upload_errors[("tgt-1", "a.pdf")] = [
        PlatformAPIError("bad", status_code=400, body="bad")
    ]
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.failed == 1
    assert result.created == 0
    assert len(report.failed_files) == 1
    entry = report.failed_files[0]
    assert entry["file_name"] == "a.pdf"
    assert "upload" in entry["error"]


@pytest.mark.parametrize(
    "mime,raw",
    [
        ("text/csv", "name,age\nalice,30"),
        ("text/plain", "plain old text"),
    ],
)
def test_text_mimes_round_trip_as_utf8(mime, raw):
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("data")]},
        file_payloads={("src-1", "data"): _text_payload(raw, mime=mime)},
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)
    report = CloneReport()

    result = FilesPhase(ctx).run(report)

    assert result.created == 1
    upload = tgt.uploaded[0]
    assert upload["data"] == raw.encode("utf-8")
    assert upload["mime_type"] == mime


def test_default_doc_mirrors_source_selection_by_filename():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("a.pdf"), _doc("b.pdf")]},
        file_payloads={
            ("src-1", "a.pdf"): _pdf_payload(b"A"),
            ("src-1", "b.pdf"): _pdf_payload(b"B"),
        },
        # Source's selected doc is b.pdf (document_id="src-b.pdf").
        tools=[{"tool_id": "src-1", "tool_name": "demo", "output": "src-b.pdf"}],
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    FilesPhase(ctx).run(CloneReport())

    # Target's CustomTool.output now points at b.pdf's new target doc id.
    tgt_tool = next(t for t in tgt._tools if t["tool_id"] == "tgt-1")
    output_id = tgt_tool["output"]
    b_upload = next(d for d in tgt._documents["tgt-1"] if d["document_name"] == "b.pdf")
    assert output_id == b_upload["document_id"]


def test_default_doc_falls_back_to_first_when_source_has_none():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("a.pdf")]},
        file_payloads={("src-1", "a.pdf"): _pdf_payload(b"A")},
        # Source has no output set.
        tools=[{"tool_id": "src-1", "tool_name": "demo"}],
    )
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT, tools=[{"tool_id": "tgt-1", "tool_name": "demo"}]
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    FilesPhase(ctx).run(CloneReport())

    tgt_tool = next(t for t in tgt._tools if t["tool_id"] == "tgt-1")
    a_upload = next(d for d in tgt._documents["tgt-1"] if d["document_name"] == "a.pdf")
    assert tgt_tool["output"] == a_upload["document_id"]


def test_default_doc_preserves_existing_target_choice():
    src = FakeClient(
        endpoint=SRC_ENDPOINT,
        documents={"src-1": [_doc("a.pdf")]},
        file_payloads={("src-1", "a.pdf"): _pdf_payload(b"A")},
        tools=[{"tool_id": "src-1", "tool_name": "demo", "output": "src-a.pdf"}],
    )
    # Operator already picked a doc on target — re-run must not clobber.
    tgt = FakeClient(
        endpoint=TGT_ENDPOINT,
        tools=[{"tool_id": "tgt-1", "tool_name": "demo", "output": "operator-pick"}],
    )
    remap = RemapTable()
    remap.record("custom_tool", "src-1", "tgt-1")
    ctx = _ctx(src, tgt, remap=remap)

    FilesPhase(ctx).run(CloneReport())

    tgt_tool = next(t for t in tgt._tools if t["tool_id"] == "tgt-1")
    assert tgt_tool["output"] == "operator-pick"
