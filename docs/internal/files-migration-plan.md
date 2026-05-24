# Files-Migration Phase — Implementation Plan

**Status:** draft, revised 2026-05-24 (post product-owner review). Not a KB doc. Authoritative runbook for the SDK implementor agent (`odm-sdk-impl`) to land the `files` phase end-to-end.

**Constraints from product owner:**
- **No new backend Platform API endpoints.** Use what exists today.
- **No storage-backend direct copy** (per-provider CLI + credentials handoff is too painful to maintain across customer environments).
- **No signed-URL relay** (would depend on new BE endpoints).
- Backend changes limited to non-API refactors (streaming write helper) that benefit the FE simultaneously.

**Net scope: Platform-API uploads only. Oversize files are reported, not aborted on — operator handles those via UI re-upload.**

---

## 0. What this phase moves

Prompt Studio document files only. The user-uploaded test corpus per `CustomTool`. Storage path on both ends:

```
{PERMANENT_REMOTE_STORAGE}/{REMOTE_PROMPT_STUDIO_FILE_PATH}/{org_id}/{user_id}/{tool_id}/<file>
```

Server-generated subdirs (`extract/`, `summarize/`, `converted/`) are **out of scope** — regenerated on target on first index/summarize/preview. Don't migrate them.

`DocumentManager` rows (`prompt_studio_document_manager_v2.DocumentManager`) are the DB-side mirror of the files. **The Platform API upload endpoint creates the `DocumentManager` row as a side effect of the file upload itself** — see §10a. `CustomToolPhase` does not create them.

## 1. Where the phase sits

Dependency order:

```
… → custom_tool → files → workflow → tool_instance → workflow_endpoint → api_deployment → pipeline
```

Runs after `custom_tool` (which mints the `src_tool_id → tgt_tool_id` remap and creates the CustomTool + ProfileManager + prompts on target) and before `workflow` (workflows can reference indexed prompt outputs, but `workflow` creation itself doesn't need the bytes present).

## 2. Strategy dispatch

Selectable via `MigrationOptions.file_strategy`:

| Value | Behavior |
|-------|----------|
| `"platform_api"` (default) | Uploads through existing Platform API endpoints, capped per file. Files over cap are reported, not aborted on. |
| `"skip"` | Operator re-uploads manually via UI on target; SDK does pure metadata migration. |

**Sizing rationale (US prod data, 2026-05-24):** 18,890 orgs with files, p99=14 files/org, max external=529 (buildwithstudio). Moody's biggest single user-org=99 files. At ~3 sec per file roundtrip, Moody's worst case = ~5 min, buildwithstudio-class = ~25 min. Sequential through the Platform API is in budget. File sizes not in DB — sampling recommended but not blocking; cap is the safety valve.

## 3. Default upload flow

### Mechanism

Use the existing endpoints exactly as they are:

- **Download from source:** `GET /api/v1/<org>/prompt-studio/file/<src_tool_id>/?file_name=<name>` — returns `{"data": <base64 or text>, "mime_type": "..."}` per `PromptStudioFileHelper.fetch_file_contents`.
- **Upload to target:** `POST /api/v1/<org>/prompt-studio/file/<tgt_tool_id>/` with multipart body — see `upload_for_ide` signature.
- **List target files for idempotency:** `GET /api/v1/<org>/prompt-document/?tool_id=<tgt_tool_id>` — returns `[{document_id, document_name, tool}, ...]`; the `document_name` field is what the pre-check matches against.

### Memory + size discipline

- **Hard per-file cap.** Default `max_file_size = 25 * 1024 * 1024` (25 MB). Above → skip that file, log to `MigrationReport.oversize_files`, continue siblings. Operator can override via CLI flag (`--max-file-size`).
- **Cap rationale.** 10 MB is too tight (a 50-page scanned PDF can be 15-30 MB); 50 MB risks cloud-worker memory pressure given the base64+JSON-wrap overhead on download (~3x the file size in peak memory). 25 MB covers typical Prompt Studio test docs while staying within a safe worker memory budget.
- **No reliable pre-flight size check.** Existing endpoints don't expose size separately from full payload. Workaround: enforce cap at request time after download — the SDK only knows the size after it has the bytes in memory. Acceptable given the cap is small enough that a single oversize-download spike is bounded.
- **Concurrency = 1** per phase. Do not raise — single-file-at-a-time is what keeps the cloud worker hold bounded.
- **Retry** transient HTTP errors (5xx, ConnectionError, Timeout) with exponential backoff, max 3 attempts.
- **No body logging** in either direction.

### Idempotency caching (important for re-run cost)

Fetch the target tool's file list **once per tool**, not per-file. With 529-file corpora (buildwithstudio scale), per-entity name lookups would balloon to 529 list calls just for skip-checks. Pattern:

```python
def migrate_tool_files(src_tool_id, tgt_tool_id):
    tgt_filenames = set(target.list_tool_filenames(tgt_tool_id))  # 1 call
    src_files = source.list_tool_filenames(src_tool_id)            # 1 call
    for fname in src_files:
        if fname in tgt_filenames:
            report.skipped_existing += 1
            continue
        migrate_file(src_tool_id, tgt_tool_id, fname, tgt_filenames)
```

This keeps re-run cost at ~2 HTTP calls per tool regardless of file count. Moody's full re-run (10 user-orgs × ~10 tools × 2 calls) ≈ 200 quick HTTP calls ≈ 10-20 sec total for the files phase.

### Per-file flow

```python
def migrate_file(src_tool_id, tgt_tool_id, file_name):
    # idempotency: skip if name already present on target
    if file_name in target_filenames_for(tgt_tool_id):
        report.skipped += 1
        return

    # download — full file in memory (existing endpoint constraint)
    resp = source.platform_api.get(
        f"/{src_org_slug}/prompt-studio/file/{src_tool_id}/",
        params={"file_name": file_name},
    )
    payload = resp.json()
    mime = payload["mime_type"]
    data_field = payload["data"]

    if mime == "application/pdf":
        raw = base64.b64decode(data_field)
    elif mime in ("text/plain", "text/csv"):
        raw = data_field.encode("utf-8")
    elif mime.startswith("application/vnd.ms-excel") or mime.startswith("application/vnd.openxmlformats"):
        raw = base64.b64decode(data_field)  # verify against helper's actual branch
    else:
        report.warnings.append(f"{file_name}: unknown mime '{mime}', skipping")
        return

    if len(raw) > options.max_file_size:
        report.oversize_files.append({
            "tool_id": tgt_tool_id,
            "tool_name": tgt_tool_name,
            "file_name": file_name,
            "size_bytes": len(raw),
            "cap_bytes": options.max_file_size,
        })
        return  # oversize → operator handles via UI re-upload

    # upload as multipart
    target.platform_api.post(
        f"/{tgt_org_slug}/prompt-studio/file/{tgt_tool_id}/",
        files={"file": (file_name, raw, mime)},
    )
    report.uploaded += 1
```

Verify mime-branch coverage against the actual `fetch_file_contents` helper (lines 167-188 of `prompt_studio_file_helper.py`) and extend if new branches landed.

### Cloud safety check

Before phase starts, log:

```
Files phase about to run via Platform API:
  source: <base_url>   target: <base_url>
  tools: <N>           cap: 25 MB/file
  estimated cloud-worker hold: ~1 worker × <duration estimate>
```

Operator can abort here if running against cloud during peak hours.

### Idempotency

Name-based, listed-target side. No hash check.

## 4. Skip mode

Pure metadata migration. Phase prints, per migrated tool:

```
files: skipped (--skip-files). Documents must be re-uploaded on target via the IDE.
  Navigate to each migrated tool on the target deployment and re-upload via the file manager pane.

  - tool 'invoice_extractor' (tgt_id=abc-123): 12 files expected
      sample.pdf, contract_q1.pdf, ...
  - tool 'receipt_classifier' (tgt_id=def-456): 3 files expected
      receipt1.pdf, receipt2.pdf, receipt3.pdf
```

**No "destination path" is needed** — the operator clicks "upload" in the target UI's file manager, picks the file from their local disk, and the backend's `upload_for_ide` constructs the storage path from the calling user's session and the target `tool_id`. The operator never touches storage paths.

**What the operator needs in their hand to do skip-files re-uploads:** the source bytes themselves. Options:
- They already have local copies (they originally uploaded these from their own machine).
- They download from source deployment UI one file at a time before re-uploading to target. Painful at >10-20 files; this is why uploads via Platform API are the default.

`MigrationReport.skipped_files` carries the full list `[{tool_id, tool_name, file_name, source_org_slug, source_tool_id}, ...]` so external tooling (or a future helper script) can drive a download-then-upload loop using the source deployment's credentials.

## 4a. Oversize file handling

When a file exceeds `max_file_size` during the default flow, the SDK does **not** abort. It:

1. Logs the file under `MigrationReport.oversize_files` with `{tool_id, tool_name, file_name, size_bytes, cap_bytes}`.
2. Continues with sibling files in the same tool.
3. At end-of-phase, prints a "files requiring manual upload" section listing the oversize subset.

A corpus where 95% of files fit under cap therefore gets 95% auto-migrated and 5% surfaced for manual UI re-upload, in one run.

## 6. Backend buffering side-quest (separate commit, not blocking the phase)

The product owner has approved buffering the upload side in the BE because it benefits the FE concurrent-upload path too. **Constraint:** no new endpoints, no behavior change. Just internal refactor.

### Change

In `backend/utils/file_storage/helpers/prompt_studio_file_helper.py:upload_for_ide`, replace:

```python
fs_instance.write(
    path=file_path,
    mode="wb",
    data=file_data if isinstance(file_data, bytes) else file_data.read(),
)
```

with chunked streaming when `file_data` is an UploadedFile:

```python
if isinstance(file_data, bytes):
    fs_instance.write(path=file_path, mode="wb", data=file_data)
else:
    with fs_instance.open(file_path, mode="wb", block_size=8 * 1024 * 1024) as out:
        for chunk in file_data.chunks(chunk_size=8 * 1024 * 1024):
            out.write(chunk)
```

Apply identical change to `upload_converted_for_ide` (same shape, different path).

### Regression risks to verify

1. **`block_size` must be set explicitly** — fsspec defaults vary per provider (GCS, S3, MinIO, Azure). Without it the implementation may buffer to memory until much larger thresholds.
2. **Partial-write cleanup on failure.** Wrap the streaming write in `try`/`except`; on exception, call the underlying multipart-abort if available (`fs.cancel(...)` for GCS resumable; `abort_multipart_upload` for S3 — exposed via `fs.fs.abort_multipart_upload` on s3fs). Document if a provider doesn't support clean abort; aged-out incomplete multipart uploads cost storage money.
3. **MIME detection.** `fetch_file_contents` does `fs.mime_type(...)` which is already partial-read-capable but tests should cover edge file types (Office docs, oddly-headered PDFs).
4. **`isinstance(file_data, bytes)` branch preserved** — some callers pass raw bytes; that path stays single-shot.
5. **Test coverage.** Add streaming-specific tests under `unstract/sdk1/tests/file_storage/` mirroring existing single-shot tests. Confirm peak memory bounded via `tracemalloc` snapshot in test.
6. **No change to `fetch_contents_ide`.** Constraint says no new BE APIs and FE consumes the existing response shape (base64 in JSON). Leave it alone.

### Out of scope for this commit

- Download streaming — would change `fetch_contents_ide` contract; product owner ruled out.
- New raw-download endpoint — same constraint.
- Resumability on the upload side — would also require new endpoint surface.

## 7. SDK package layout

```
src/unstract/migration/
  ├── phases/
  │   └── files.py             # FilesPhase: upload via Platform API with oversize reporting, or skip
  ├── file_transport/
  │   ├── __init__.py
  │   └── platform_api.py      # download (base64-decode) + upload (multipart) using existing endpoints
  └── ... (existing phases unchanged)
```

No direct-storage-copy or signed-URL transport modules — those approaches are out of scope.

## 8. `MigrationOptions` additions

```python
@dataclass
class MigrationOptions:
    # ... existing fields ...
    file_strategy: Literal["platform_api", "skip"] = "platform_api"
    max_file_size: int = 25 * 1024 * 1024  # 25 MB; oversize files are reported, not uploaded
```

CLI flags:

```
--file-strategy {platform_api,skip}    # default: platform_api
--max-file-size 25MB                    # accepts human-readable sizes
--skip-files                            # alias for --file-strategy=skip
```

No `auto` dispatch needed — only one byte-moving strategy exists.

## 9. Test plan

| Test | Mode | What it proves |
|------|------|----------------|
| 5 MB PDF via Platform API | default | round-trip base64+multipart works against existing helpers |
| 30 MB PDF via Platform API | default | cap fires; entry appears in `oversize_files`; sibling files continue |
| Tool with 10 files (mix of small + 1 oversize) | default | 9 migrated, 1 reported for manual upload, run exits 0 |
| Re-run after success | default | name-based idempotency: target file present → skip |
| `--skip-files` | skip | pure metadata migration; report lists tools + filenames (no storage paths) |
| `MigrationReport.skipped_files` schema | skip | shape matches `[{tool_id, tool_name, file_name, source_org_slug, source_tool_id}]` |
| `MigrationReport.oversize_files` schema | default | shape matches `[{tool_id, tool_name, file_name, size_bytes, cap_bytes}]` |
| Streaming upload backend refactor: 100 MB PDF via UI | (BE) | peak RSS bounded to ~chunk_size; file is byte-identical |
| Streaming upload + network failure mid-stream | (BE) | partial multipart aborted; no orphan upload in bucket |
| Default mode concurrent with FE user uploading | default | both succeed; no worker starvation observable |
| Moody's-scale run (~99 files, all under cap) | default | completes in <10 min; no failures; report counts match input |

## 10. Acceptance

- Adapter, connector, tag, custom_tool, prompts, profile_managers, prompt_registry phases unchanged.
- New `files` phase wired into `migrate()` orchestrator after `custom_tool`.
- Local smoke: fresh target org, run default mode against a tool with 3 small PDFs. Re-run: 3 skips, 0 failures.
- Local smoke: same scenario with one 30 MB PDF added → 3 uploaded, 1 in `oversize_files`, exit 0.
- Local smoke: `--skip-files` against same setup → 0 uploaded, all 4 listed in `skipped_files` with tool + filename.
- BE refactor commit (separate from SDK commit) lands on `feat/org-migration-platform-api-gaps`. SDK commit lands on `feat/org-migration`.
- `MigrationReport` exposes: `uploaded_files`, `skipped_files`, `oversize_files`, `failed_files` (each a typed list with tool_id + tool_name + file_name minimum).

## 10a. Idempotency model — pre-check is load-bearing for correctness

`upload_for_ide` (`views.py:1009`) is **not idempotent**, and in fact actively errors on retries. Order of operations:

1. `PromptStudioFileHelper.upload_for_ide(...)` — write file to storage. Overwrites on collision (storage-idempotent).
2. `PromptStudioDocumentHelper.create(tool_id, document_name)` — unconditional `DocumentManager.objects.create`.

The `DocumentManager` model carries `UniqueConstraint(fields=["document_name", "tool"])`. So calling `upload_for_ide` twice for the same `(tool, filename)` overwrites the file in storage, then raises `IntegrityError` on the second create — propagates as 500. **The SDK MUST pre-check; otherwise re-runs error out.**

For files specifically, the pattern is:

```python
def migrate_tool_files(src_tool_id, tgt_tool_id):
    tgt_filenames = set(target.list_dm_rows(tgt_tool_id))
    for src_fname in source.list_dm_rows(src_tool_id):
        if src_fname in tgt_filenames:
            continue
        upload_file(src_tool_id, tgt_tool_id, src_fname)
```

This guarantees the SDK never invokes the non-idempotent upload twice. The duplicate-DM-row outcome (well, the 500 — see above) only materializes if **something other than the SDK** invokes the upload endpoint with a name the SDK has already migrated.

### Important difference from earlier draft of this plan

This plan previously claimed `CustomToolPhase` creates `DocumentManager` rows on target. **It does not.** `import_project` only creates `CustomTool` + `ProfileManager` + prompts; DM rows are only created by `upload_for_ide`. Therefore:

- Before the files phase runs on a fresh target tool, the DM list is empty. Pre-check returns ∅. SDK uploads all files. ✅
- On re-run, pre-check returns the SDK's own previously-created rows. SDK skips them. ✅
- For an oversize / skipped file the operator re-uploads via UI: no SDK DM row exists for that filename → UI creates a single row, no duplicate. ✅

The duplicate-DM-row problem from prior plan revisions is **not reachable** through SDK flows. It only manifests if a user uploads via UI mid-migration (between the SDK's list call and its upload call for that same filename) — mitigation is to run migrations in low-activity windows.

### Crash semantics

The file-first-then-DM order is fortunate:

| Failure point | Target state | Re-run outcome |
|---------------|--------------|----------------|
| File write fails | No file, no DM row | Pre-check sees no DM → retry → clean |
| File written, DM create fails | File on disk, no DM row | Pre-check sees no DM → retry → file overwritten, DM created. **Self-healing** |
| Both succeed, SDK dies | File + DM both present | Pre-check sees DM → skip. Correct |
| Both succeed, network blip on response | File + DM both present, SDK saw error | Same. SDK reconciles via pre-check, not via its own ack |

The opposite order (DM first, file second) would leave a "ghost DM row pointing at nothing" state that the pre-check couldn't distinguish from a real one. We get lucky here.

### Operator UI re-upload caveat

When the operator manually re-uploads a file via the target UI to fill in something the SDK skipped or marked oversize, the UI hits the same non-idempotent endpoint. Because the SDK never created a DM row for that filename in those skipped cases, the UI's create succeeds cleanly — no duplicate.

The only failure pattern: operator re-uploads a file the SDK already migrated (i.e. a file already on disk and in the DM table). The unique constraint will reject this with a 500. UI surfaces an error; the operator has to delete the existing row first via UI, then re-upload.

### Optional backend cleanup (out of scope)

`PromptStudioDocumentHelper.create` could be changed to `get_or_create(tool, document_name)` — eliminates the 500 on re-upload, benefits UI behavior too. **Not required by the SDK** (the pre-check makes the SDK safe regardless) and not blocking; queue as a follow-up if UI ergonomics complaints arrive.

Document explicitly in the README's "What if files aren't on disk?" and "Files phase specifics" sections.

## 11. What to push back on if encountered

- Any request to add a new BE endpoint → confirm with product owner first; the constraint is "no new APIs".
- Any request to change `fetch_contents_ide` shape → same; FE consumes the base64+JSON envelope today.
- Any request to raise `max_file_size` above 50 MB → confirm cloud-worker RAM budget first; default 25 MB exists because base64 round-trip × 2 already runs ~3x file size in peak worker memory.
- Files larger than 25 MB common in the customer's corpus → that's expected to be a tail; operator handles via UI re-upload. If the tail is the majority, escalate — may need to revisit the no-new-endpoints constraint for a streaming download endpoint.
- Any request to copy files directly between storage backends or via signed URLs → ruled out: per-provider CLI maintenance burden, and signed URLs would need new BE endpoints.

## 12. References

- `backend/utils/file_storage/helpers/prompt_studio_file_helper.py` — `upload_for_ide`, `fetch_file_contents` (note: `fetch_contents_ide` returns base64-wrapped raw bytes for PDF, not LLMW-extracted text — extracted text lives under `extract/` subdir, written by a different code path)
- `backend/prompt_studio/prompt_studio_core_v2/urls.py` — `prompt_studio_file` route
- `backend/prompt_studio/prompt_studio_document_manager_v2/` — `DocumentManager` model
- Branch: backend changes → `feat/org-migration-platform-api-gaps`; SDK changes → `feat/org-migration`
