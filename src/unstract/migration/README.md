# Org-to-Org Migration

Move an Unstract organization's resources from one deployment to another over the Platform API.

What gets carried over: adapters, connectors, custom tools, prompts, profiles, workflows, tool instances, workflow endpoints, tags, API deployments, pipelines, and Prompt Studio document files.

## Quickstart

```bash
UNSTRACT_SRC_PLATFORM_KEY=src_pk_...  \
UNSTRACT_TGT_PLATFORM_KEY=tgt_pk_...  \
uv run python -m unstract.migration migrate \
  --source-url https://source.example.com \
  --source-org my-source-org \
  --target-url https://target.example.com \
  --target-org my-target-org
```

You need an **org admin Platform API key** for both ends.

> [!WARNING]
> Both keys grant full read on source and full write on target. Run from a trusted machine and **rotate both keys after the migration completes**.

## How it works

The tool walks resources in dependency order. Each phase migrates one type and remembers the new IDs so later phases can rewrite references before posting.

```
1. adapter           7. republish_tool
2. connector         8. files                  (Prompt Studio documents)
3. tag               9. workflow
4. custom_tool      10. tool_instance
5. profile_manager  11. workflow_endpoint
6. prompt           12. api_deployment
                    13. pipeline
```

## Re-runs are safe

Stop the script mid-run, fix what broke, run the same command again — it picks up where it left off. The tool checks the target by name before creating anything; resources that already exist are reused.

A clean re-run after a successful migration does no writes and finishes in 1–2 minutes (a first run on a moderate corpus takes 7–10).

There is no resume flag and no state file. The target *is* the state — if you delete a resource on the target between runs, the next run recreates it.

## If something fails partway

Each resource is its own request and its own transaction. There is no all-or-nothing rollback for a phase.

1. Read the printed `MigrationReport` — it lists completed phases and the entity that failed.
2. Fix the underlying issue.
3. Re-run the same command.

> [!NOTE]
> API deployments and pipelines get a **new API key minted on the target**. Downstream consumers must be updated with the new key.

> [!NOTE]
> Pipelines are created **paused** on the target so scheduled runs don't fire during cut-over. Unpause them once you're ready. Override with `--no-pipelines-paused`.

## Files

The Prompt Studio document corpus is the only thing with actual bytes on disk. Default strategy downloads each file from source and uploads to target, one at a time, capped at 25 MB per file by default.

| `--file-strategy` | Behavior |
|-------------------|----------|
| `platform_api` (default) | Transfer each file via the Platform API. Files over `--max-file-size` are skipped and listed at the end for manual re-upload. |
| `skip` | Don't transfer any files. Document records are still created on the target. Equivalent to `--skip-files`. |

> [!WARNING]
> If you run migrations while users are actively uploading to the same source org, you can end up with duplicate file records on the target. **Run migrations in low-activity windows.**

> [!NOTE]
> If a file is missing on disk (skipped, oversize, or a mid-run crash), the platform stays usable. Only operations that touch that specific file (preview, index, prompt run) will error. Re-upload missing files through the UI.

## What you'll see in the report

`MigrationReport` prints at the end with:

- Per-phase counts: `created`, `adopted` (already existed), `failed`
- `oversize_files` — files skipped because they exceeded the cap
- `skipped_files` — files not transferred under `--file-strategy=skip`
- `failed_files` — files the upload itself failed on
- A source-to-target UUID map for every migrated resource

## CLI reference

### Environment

| Var | Required | Purpose |
|-----|----------|---------|
| `UNSTRACT_SRC_PLATFORM_KEY` | yes | Source org admin Platform API key |
| `UNSTRACT_TGT_PLATFORM_KEY` | yes | Target org admin Platform API key |

### Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--source-url` / `--target-url` | — | Base URLs of both deployments |
| `--source-org` / `--target-org` | — | Org slugs |
| `--api-prefix` | `api/v1` | URL prefix for the Platform API |
| `--include` / `--exclude` | all / none | Comma-separated phase names |
| `--dry-run` | off | List actions without writing |
| `--on-name-conflict` | `adopt` | `adopt` reuses existing target resources; `abort` stops on conflict |
| `--file-strategy` | `platform_api` | `platform_api` or `skip` |
| `--max-file-size` | `25MB` | Per-file cap |
| `--skip-files` | off | Alias for `--file-strategy=skip` |
| `--pipelines-paused` | on | Create pipelines paused on target |
| `--verbose` | off | Per-entity log lines |

## Things to keep in mind

- **Adapter and connector secrets are carried verbatim.** They never appear in logs, but they do travel over the wire to the target — both deployments must be ones you trust.
- **UUIDs are not preserved.** Every target resource gets a fresh UUID. References between resources are rewritten automatically.
- **Direct storage-bucket copy isn't supported.** Files always go through the Platform API.
- **Run from a trusted machine.** Both API keys are loaded as environment variables.
