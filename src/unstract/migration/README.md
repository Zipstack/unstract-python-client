# `unstract.migration` — Org-to-Org Data Migration

SDK subpackage that lifts an Unstract organization's configured resources from one deployment into another using existing Platform API endpoints. Adapters, connectors, custom tools, prompts, profiles, workflows, tool instances, workflow endpoints, tags, API deployments, pipelines, and Prompt Studio document files.

## Quickstart

```bash
UNSTRACT_SRC_PLATFORM_KEY=src_pk_...  \
UNSTRACT_TGT_PLATFORM_KEY=tgt_pk_...  \
uv run python -m unstract.migration migrate \
  --source-url https://us.unstract.com \
  --source-org my-source-org \
  --target-url https://us.unstract.com \
  --target-org my-target-org
```

Both keys must be **org admin Platform API keys**. Run from a trusted machine.

## How it works

The SDK orchestrates Platform API calls in a strict dependency order. Each phase migrates one resource type and feeds a remap table (`source_uuid → target_uuid`) that later phases consume to rewrite embedded references before POST.

Phase order:

```
1. adapter           7. republish_tool
2. connector         8. files                  (Prompt Studio document corpus)
3. tag               9. workflow
4. custom_tool      10. tool_instance
5. profile_manager  11. workflow_endpoint
6. prompt           12. api_deployment
                    13. pipeline
```

Phases 4–7 are composite (run together under `CustomToolPhase`).

## Failure semantics — important

### DB writes are committed per-resource, not per-phase

Each POST is a separate Django request and a separate transaction on the target. There is **no all-or-nothing transaction wrapping a phase**. Consequences:

- If a phase fails on the Nth entity, entities 1..N-1 are present on target. Entity N is rolled back (its own transaction). Entities N+1..M are never attempted.
- Side-effects that ride on POSTs (API keys auto-minted for API deployments and pipelines, PeriodicTask rows for scheduled pipelines, `DocumentManager` rows for tool documents) are persisted alongside their parent — same per-resource atomicity.
- The in-memory `RemapTable` is process state and is lost on crash. Re-run rebuilds it via Layer 2 idempotency.

### Re-runs are idempotent and cheap

**None of the Platform API write endpoints are naturally idempotent** — POSTing the same adapter / connector / workflow / file twice produces two target rows. The SDK works around this with a uniform pattern: **pre-check the target by name before POSTing.**

Every phase:
- Lists target by name filter (or by parent tool, for files) — one call per phase.
- For each source entity: if already present on target, record `src_uuid → tgt_uuid` in the in-memory remap and skip the POST. If missing, per-id GET on source for the full payload, remap UUIDs, POST.

The endpoint stays non-idempotent; the SDK guarantees idempotency by **not invoking the endpoint twice**.

On a clean re-run after a fully-successful migration, no POSTs fire. Cost reduces to one list call per phase per tool. Typical re-run time: 1–2 minutes for a moderate corpus vs. 7–10 minutes for the first run.

On a re-run after a partial-failure crash, completed phases skip-everything; the crashed phase resumes from the first missing entity.

#### Files phase specifics

The upload endpoint (`upload_for_ide`) writes the file to storage **first**, then creates the `DocumentManager` row. Both are unconditional — the endpoint has no upsert. Two consequences:

- Partial-failure between "file written" and "DM row created" leaves a file with no DM row. The SDK's pre-check looks at DM rows (filenames); seeing no row, it retries the upload, the file is overwritten (storage-idempotent), and the DM row is created. **Self-healing on re-run.**
- Once both succeed, the SDK's pre-check on the next run sees the DM row and skips the upload call entirely. No duplicate DM row.

The one realistic case where the SDK's pre-check can be defeated is **concurrent UI upload mid-migration**: a user uploading the same filename through the IDE after the SDK already listed target filenames. The SDK then uploads anyway, creating a duplicate DM row. **Mitigation: run migrations in low-activity windows.**

### Files phase is the exception worth knowing about

Files are uploaded per-file, one at a time. Each upload is its own request. Failure semantics match the metadata phases (per-file commit, no all-or-nothing). But two extra wrinkles:

- **Oversize files** (above `--max-file-size`, default 25 MB) are not uploaded; they are recorded in `MigrationReport.oversize_files` and listed at end-of-phase for manual UI re-upload. The run does not abort.
- **If the files phase fails or is skipped**, the target has `CustomTool` + `DocumentManager` rows but no actual files in storage. The platform stays usable globally; per-file operations (preview, index, prompt-run) on missing files error cleanly. Users can re-upload missing files via the target UI's file manager. The platform doesn't crash.

See [What if files aren't on disk?](#what-if-files-arent-on-disk) for details.

### How to recover from a mid-failure crash

1. Read the printed `MigrationReport` — completed phases + the entity that failed.
2. Fix the underlying issue (network, permissions, oversize payload, etc.).
3. Re-run the same command. The SDK picks up where it left off.

There is no `--resume-from` flag and no state file. The target *is* the state.

## What gets migrated

| Resource | Notes |
|----------|-------|
| Adapters | Including decrypted `adapter_metadata` (carries secrets verbatim — same surface the FE already consumes) |
| Connectors | Same secrets posture |
| Tags | Per-org |
| Custom tools | + nested: profile_managers, prompts, document_manager rows |
| Prompt registry | Re-published on target via `update_or_create` (no manual carry) |
| Files | Prompt Studio document corpus per tool — see [Files phase](#files-phase) |
| Workflows | Workflow_name remapped |
| Tool instances | v1 assumes ≤1 per workflow |
| Workflow endpoints | Connector references remapped |
| API deployments | New API key minted on target (consumer keys regenerate; document for downstream consumers) |
| Pipelines | New API key + PeriodicTask auto-minted server-side. Default state: paused (`active=false`) — SDK PATCHes immediately after POST to avoid cron firing on a half-cut-over org |

## Files phase

The only resource type with bytes-on-disk that migrate. Storage path on both ends:

```
{PERMANENT_REMOTE_STORAGE}/{REMOTE_PROMPT_STUDIO_FILE_PATH}/{org_id}/{user_id}/{tool_id}/<file>
```

Server-generated subdirs (`extract/`, `summarize/`, `converted/`) are **out of scope** — regenerated on target on first index/summarize/preview.

### Strategy

Two modes; default is `platform_api`.

| `--file-strategy` | Behavior |
|-------------------|----------|
| `platform_api` (default) | Download each file via existing `fetch_contents_ide` endpoint, upload via `upload_for_ide`. Cap per file = `--max-file-size` (default 25 MB). Files over cap are reported for manual re-upload, not aborted. Concurrency = 1. |
| `skip` | No bytes touched. `DocumentManager` rows present on target (from `CustomToolPhase`), files missing on disk. Report lists every expected filename for manual UI re-upload. Equivalent: `--skip-files`. |

### What if files aren't on disk?

After a `skip`, after oversize-file reporting in `platform_api` mode, or after a mid-failure crash before bytes were transferred, the target has `DocumentManager` rows that reference files not present in storage. **The platform stays usable globally.** Specifically:

- Tool/workflow/deployment/pipeline listing and navigation: works.
- Opening any CustomTool in Prompt Studio: works.
- Per-file preview pane: errors with a `FileNotFoundError`-derived 500 (no explicit handler in the view today). UI shows an error.
- Index document / run prompt against missing file: errors cleanly (explicit handler).
- Re-upload via UI: works — restores the file. **Caveat:** the upload endpoint is not idempotent at the DB layer; it creates a new `DocumentManager` row unconditionally. If migration already created a DM row for that filename (it does via `CustomToolPhase`), the UI re-upload produces a second DM row pointing at the same (overwritten) file. UI will list the file twice. Delete the stale row via UI first, then re-upload, to avoid duplicates. The SDK itself avoids this trap by pre-checking DM rows before any upload call; the duplicate is purely a UI-side re-upload artifact.
- All other tools/workflows that have their files: unaffected.

So a partial files migration leaves users able to use the platform broadly; only the specific missing files surface errors when touched.

## Constraints and trade-offs

- **No new backend API endpoints** — files phase uses what exists today. The download path eats a ~33% base64 inflation and one-shot full-file memory on both ends. That's why the size cap is conservative.
- **Storage-backend direct copy (`gsutil rsync` / `aws s3 sync`) not supported** — per-provider CLI maintenance burden is too high.
- **No state file** — idempotency relies on target being queryable by name. If you delete a target resource between runs, the SDK recreates it on the next run.
- **No UUID preservation** — every target resource gets a freshly minted UUID. Embedded references are remapped via the in-memory `RemapTable`.

## Configuration reference

### Environment

| Var | Required | Purpose |
|-----|----------|---------|
| `UNSTRACT_SRC_PLATFORM_KEY` | yes | Source org admin Platform API key |
| `UNSTRACT_TGT_PLATFORM_KEY` | yes | Target org admin Platform API key |

### CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--source-url` / `--target-url` | — | Base URLs of both deployments |
| `--source-org` / `--target-org` | — | Org slugs |
| `--api-prefix` | `api/v1` | URL prefix; varies on cloud |
| `--include` / `--exclude` | all / none | Phase filter (comma-separated phase names) |
| `--dry-run` | off | List actions, don't POST |
| `--on-name-conflict` | `adopt` | `adopt` (skip existing) or `abort` |
| `--file-strategy` | `platform_api` | `platform_api` or `skip` |
| `--max-file-size` | `25MB` | Per-file cap for files phase |
| `--skip-files` | off | Alias for `--file-strategy=skip` |
| `--pipelines-paused` | on | Toggle the post-POST PATCH that pauses pipelines on target |
| `--verbose` | off | Per-entity log lines |

## Report shape

`MigrationReport` exposes:

- `created` / `adopted` / `failed` counts per phase
- `oversize_files: list[{tool_id, tool_name, file_name, size_bytes, cap_bytes}]`
- `skipped_files: list[{tool_id, tool_name, file_name, source_org_slug, source_tool_id}]`
- `failed_files: list[{tool_id, tool_name, file_name, error}]`
- `remap_snapshot: dict[entity_type, dict[src_uuid, tgt_uuid]]`
- A pretty-printed source-to-target UUID map at end (rich-formatted; plain-text fallback)

## Logging hygiene

- Secret values (adapter/connector metadata) are not logged.
- File request/response bodies are not logged.
- Per-entity log lines format: `src=<uuid> -> tgt=<uuid>` plus entity name + type.
- Rotate both Platform API keys after the migration completes.

## Further reading

- KB: `~/Documents/Obsidian Vault/zipstuff/org-data-migration/` (start with `INDEX.md`)
- Implementation plan for the files phase: `docs/internal/files-migration-plan.md`
