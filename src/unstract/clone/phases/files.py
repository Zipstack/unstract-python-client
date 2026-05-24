"""Migrate Prompt Studio document files (the user-uploaded test corpus).

Runs after ``CustomToolPhase`` — consumes the ``custom_tool`` remap to
know which source-tool to target-tool mapping to iterate.

Default mode (``file_strategy='platform_api'``):

1. For each ``(src_tool_id, tgt_tool_id)``, list source DM rows + target
   DM rows once each.
2. For each source filename missing on target: download from source, decode
   per mime, enforce the size cap, upload as multipart to target.
3. Oversize files → ``CloneReport.oversize_files``; mime types the
   backend can't round-trip (Excel placeholder, etc) →
   ``unsupported_files``; transport errors → ``failed_files``.

Skip mode (``file_strategy='skip'``):

- No download/upload. Source DM list is emitted into ``skipped_files`` so
  the operator knows what to re-upload manually via UI.

Concurrency is 1 per phase by design — the Platform API endpoint holds a
cloud worker for the whole upload, and uploads are not chunked on the BE
helper today. See ``docs/internal/files-clone-plan.md`` for the
sizing rationale.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import requests

from unstract.clone.exceptions import PlatformAPIError
from unstract.clone.phases.base import Phase
from unstract.clone.report import CloneReport, PhaseResult

logger = logging.getLogger(__name__)

# Mime types the BE's fetch_contents_ide endpoint round-trips losslessly.
# PDF → base64; text/plain + text/csv → utf-8 string. Excel and other
# types return a placeholder/unhandled — must be flagged for manual upload.
_BASE64_MIMES: frozenset[str] = frozenset({"application/pdf"})
_TEXT_MIMES: frozenset[str] = frozenset({"text/plain", "text/csv"})

_RETRYABLE_STATUS: frozenset[int] = frozenset({502, 503, 504})
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE_SECONDS = 1.0


class FilesPhase(Phase):
    name = "files"

    def run(self, report: CloneReport) -> PhaseResult:
        result = report.get_phase(self.name)
        tool_remap = self.ctx.remap.snapshot().get("custom_tool", {})
        if not tool_remap:
            logger.info("files phase: no custom_tool remap entries; nothing to do")
            return result

        strategy = self.ctx.options.file_strategy
        logger.info(
            "files phase: strategy=%s tools=%d cap=%d bytes",
            strategy, len(tool_remap), self.ctx.options.max_file_size,
        )

        for src_tool_id, tgt_tool_id in tool_remap.items():
            tool_name = self._lookup_tool_name(tgt_tool_id) or src_tool_id
            try:
                src_docs = self.ctx.source.list_prompt_documents(src_tool_id)
            except Exception as e:
                logger.exception(
                    "files: failed to list source DM rows for tool %s: %s",
                    tool_name, e,
                )
                result.failed += 1
                result.errors.append(
                    f"list source docs {tool_name}: {e}"
                )
                continue

            if strategy == "skip":
                self._emit_skip(src_docs, src_tool_id, tgt_tool_id, tool_name, report, result)
                continue

            self._clone_tool(
                src_tool_id, tgt_tool_id, tool_name, src_docs, report, result
            )

        return result

    def _clone_tool(
        self,
        src_tool_id: str,
        tgt_tool_id: str,
        tool_name: str,
        src_docs: list[dict[str, Any]],
        report: CloneReport,
        result: PhaseResult,
    ) -> None:
        try:
            tgt_docs = self.ctx.target.list_prompt_documents(tgt_tool_id)
        except Exception as e:
            logger.exception(
                "files: failed to list target DM rows for tool %s: %s",
                tool_name, e,
            )
            result.failed += 1
            result.errors.append(f"list target docs {tool_name}: {e}")
            return
        target_names = {d["document_name"] for d in tgt_docs}

        for doc in src_docs:
            file_name = doc.get("document_name")
            src_document_id = doc.get("document_id")
            if not file_name or not src_document_id:
                continue
            if file_name in target_names:
                result.skipped += 1
                logger.debug(
                    "files: already present on target tool=%s file=%s",
                    tool_name, file_name,
                )
                continue
            if self.ctx.options.dry_run:
                result.skipped += 1
                logger.info(
                    "[dry-run] files: would clone tool=%s file=%s",
                    tool_name, file_name,
                )
                continue
            self._clone_one_file(
                src_tool_id,
                tgt_tool_id,
                tool_name,
                file_name,
                src_document_id,
                report,
                result,
            )

        if not self.ctx.options.dry_run:
            self._ensure_default_doc(
                src_tool_id, tgt_tool_id, tool_name, src_docs
            )

    def _clone_one_file(
        self,
        src_tool_id: str,
        tgt_tool_id: str,
        tool_name: str,
        file_name: str,
        src_document_id: str,
        report: CloneReport,
        result: PhaseResult,
    ) -> None:
        try:
            payload = self._with_retry(
                lambda: self.ctx.source.download_prompt_file(
                    src_tool_id, src_document_id
                ),
                op=f"download {tool_name}/{file_name}",
            )
        except Exception as e:
            logger.exception(
                "files: download failed tool=%s file=%s: %s",
                tool_name, file_name, e,
            )
            result.failed += 1
            report.failed_files.append(
                {
                    "tool_id": tgt_tool_id,
                    "tool_name": tool_name,
                    "file_name": file_name,
                    "error": f"download: {e}",
                }
            )
            return

        mime = (payload or {}).get("mime_type") or ""
        raw = self._decode_payload(payload, mime)
        if raw is None:
            logger.warning(
                "files: unsupported mime tool=%s file=%s mime=%s",
                tool_name, file_name, mime,
            )
            report.unsupported_files.append(
                {
                    "tool_id": tgt_tool_id,
                    "tool_name": tool_name,
                    "file_name": file_name,
                    "mime_type": mime,
                }
            )
            return

        if len(raw) > self.ctx.options.max_file_size:
            report.oversize_files.append(
                {
                    "tool_id": tgt_tool_id,
                    "tool_name": tool_name,
                    "file_name": file_name,
                    "size_bytes": len(raw),
                    "cap_bytes": self.ctx.options.max_file_size,
                }
            )
            logger.info(
                "files: oversize tool=%s file=%s size=%d cap=%d",
                tool_name, file_name, len(raw), self.ctx.options.max_file_size,
            )
            return

        try:
            self._with_retry(
                lambda: self.ctx.target.upload_prompt_file(
                    tgt_tool_id, file_name, raw, mime
                ),
                op=f"upload {tool_name}/{file_name}",
            )
        except Exception as e:
            logger.exception(
                "files: upload failed tool=%s file=%s: %s",
                tool_name, file_name, e,
            )
            result.failed += 1
            report.failed_files.append(
                {
                    "tool_id": tgt_tool_id,
                    "tool_name": tool_name,
                    "file_name": file_name,
                    "error": f"upload: {e}",
                }
            )
            return

        result.created += 1
        report.uploaded_files.append(
            {
                "tool_id": tgt_tool_id,
                "tool_name": tool_name,
                "file_name": file_name,
                "size_bytes": len(raw),
                "mime_type": mime,
            }
        )
        logger.info(
            "files: uploaded tool=%s file=%s size=%d",
            tool_name, file_name, len(raw),
        )

    def _emit_skip(
        self,
        src_docs: list[dict[str, Any]],
        src_tool_id: str,
        tgt_tool_id: str,
        tool_name: str,
        report: CloneReport,
        result: PhaseResult,
    ) -> None:
        for doc in src_docs:
            file_name = doc.get("document_name")
            if not file_name:
                continue
            report.skipped_files.append(
                {
                    "tool_id": tgt_tool_id,
                    "tool_name": tool_name,
                    "file_name": file_name,
                    "source_org_slug": self.ctx.source.endpoint.organization_id,
                    "source_tool_id": src_tool_id,
                }
            )
            result.skipped += 1
        logger.info(
            "files: skip mode emitted %d filenames for tool=%s",
            len(src_docs), tool_name,
        )

    def _decode_payload(
        self, payload: dict[str, Any] | None, mime: str
    ) -> bytes | None:
        if not payload:
            return None
        data_field = payload.get("data")
        if data_field is None:
            return None
        if mime in _BASE64_MIMES:
            # data_field is base64-encoded bytes (BE wraps with b64encode).
            if isinstance(data_field, bytes):
                return base64.b64decode(data_field)
            return base64.b64decode(data_field.encode())
        if mime in _TEXT_MIMES:
            if isinstance(data_field, bytes):
                return data_field
            return data_field.encode("utf-8")
        # Excel + unhandled types: BE returned a placeholder string,
        # not real bytes. Round-trip would corrupt the file.
        return None

    def _ensure_default_doc(
        self,
        src_tool_id: str,
        tgt_tool_id: str,
        tool_name: str,
        src_docs: list[dict[str, Any]],
    ) -> None:
        """Set target ``CustomTool.output`` so the FE auto-selects a doc.

        Mirror source's chosen doc by filename when possible; fall back
        to the first available target doc. Skip if target already has
        ``output`` set — never override an operator's later choice on
        re-runs.
        """
        try:
            tgt_tool = self.ctx.target.get_custom_tool(tgt_tool_id)
        except Exception as e:
            logger.warning(
                "files: skipping default-doc set for tool=%s — fetch tgt failed: %s",
                tool_name, e,
            )
            return

        if tgt_tool.get("output"):
            logger.debug(
                "files: target tool=%s already has default doc; leaving as-is",
                tool_name,
            )
            return

        try:
            tgt_docs = self.ctx.target.list_prompt_documents(tgt_tool_id)
        except Exception as e:
            logger.warning(
                "files: skipping default-doc set for tool=%s — list tgt docs failed: %s",
                tool_name, e,
            )
            return
        if not tgt_docs:
            return

        chosen_id = self._pick_default_doc_id(
            src_tool_id, src_docs, tgt_docs, tool_name
        )
        if not chosen_id:
            return

        try:
            self.ctx.target.update_custom_tool(tgt_tool_id, {"output": chosen_id})
            logger.info(
                "files: set default doc tool=%s doc_id=%s", tool_name, chosen_id
            )
        except Exception as e:
            logger.warning(
                "files: PATCH default doc failed tool=%s: %s", tool_name, e
            )

    def _pick_default_doc_id(
        self,
        src_tool_id: str,
        src_docs: list[dict[str, Any]],
        tgt_docs: list[dict[str, Any]],
        tool_name: str,
    ) -> str | None:
        # Try mirroring the source's selection by filename. If source
        # GET fails or source has no chosen doc, fall back to the first
        # target doc so the FE doesn't render an empty selector.
        try:
            src_tool = self.ctx.source.get_custom_tool(src_tool_id)
            src_output = src_tool.get("output")
        except Exception as e:
            logger.debug(
                "files: source CustomTool fetch failed for tool=%s (%s); "
                "falling back to first target doc",
                tool_name, e,
            )
            src_output = None

        if src_output:
            src_name = next(
                (d.get("document_name") for d in src_docs
                 if d.get("document_id") == src_output),
                None,
            )
            if src_name:
                matched = next(
                    (d.get("document_id") for d in tgt_docs
                     if d.get("document_name") == src_name),
                    None,
                )
                if matched:
                    return matched

        return tgt_docs[0].get("document_id")

    def _lookup_tool_name(self, tgt_tool_id: str) -> str | None:
        # CustomToolPhase doesn't record names; fetch lazily for log clarity.
        # One call per tool is cheap relative to the per-file traffic.
        try:
            tools = self.ctx.target.list_custom_tools()
        except Exception:
            return None
        for t in tools:
            if t.get("tool_id") == tgt_tool_id:
                return t.get("tool_name")
        return None

    def _with_retry(self, fn: Any, *, op: str) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return fn()
            except PlatformAPIError as e:
                last_exc = e
                if e.status_code not in _RETRYABLE_STATUS or attempt == _MAX_RETRIES:
                    raise
                sleep = _RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "files: retry %d/%d for %s after %d: sleeping %.1fs",
                    attempt, _MAX_RETRIES, op, e.status_code, sleep,
                )
                time.sleep(sleep)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt == _MAX_RETRIES:
                    raise
                sleep = _RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "files: retry %d/%d for %s after %s: sleeping %.1fs",
                    attempt, _MAX_RETRIES, op, type(e).__name__, sleep,
                )
                time.sleep(sleep)
        assert last_exc is not None
        raise last_exc
