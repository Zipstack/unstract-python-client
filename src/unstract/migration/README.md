# Org-to-Org Migration

Move an Unstract organization's configured resources from one deployment to another (or between two orgs on the same deployment).

Carried over: adapters, connectors, custom tools, prompts, profiles, workflows, tool instances, workflow endpoints, tags, API deployments, pipelines, and Prompt Studio document files.

> **Full documentation, behavior notes, CLI reference, and sample report:**
> https://docs.unstract.com/unstract/unstract_platform/api_documentation/versions/v1-org-migration/

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

Both keys must be **org admin Platform API keys**.

> [!WARNING]
> Both keys grant broad access. Run from a trusted machine and rotate both keys after the migration completes.

## Re-runs are safe

If a phase fails partway, fix the cause and re-run the same command. Resources already on the target are detected by name and reused. There is no `--resume-from` flag — the target is the state.

## Files

The Prompt Studio document corpus is the only resource type with bytes on disk. Default cap per file is 25 MB; oversize files are reported for manual re-upload. Use `--skip-files` to skip bytes entirely (document records are still created).

> [!WARNING]
> Run migrations during low-activity windows. Concurrent uploads to the source org during a migration can create duplicate file records on the target.

See the [public docs](https://docs.unstract.com/unstract/unstract_platform/api_documentation/versions/v1-org-migration/) for the full flag list, behavioral notes, and the format of the end-of-run report.
