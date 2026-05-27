# Cloning Organizations

> [!NOTE]
> **Users are not cloned.** Two reasons:
> - The same user may not need access in every environment.
> - The same user may hold different roles across environments.
>
> **Groups _will_ be cloned** (upcoming — not yet implemented). Once available, an admin can add the right users to each group per environment.

Clone an Unstract organization's configured resources into another organization (same deployment or different). Useful for environment promotion (DEV → QA → PROD) and for spinning up a fresh org from a known-good baseline.

Cloned resources: adapters, connectors, custom tools, prompts, profiles, workflows, tool instances, workflow endpoints, tags, API deployments, pipelines, and Prompt Studio document files. The source org is left untouched.

> **Full documentation, behavior notes, CLI reference, and sample report:**
> https://docs.unstract.com/unstract/unstract_platform/api_documentation/versions/cloning-orgs/

## Install

From a clone of this repository:

```bash
uv sync --all-extras
```

This pulls in the `clone` extra (`click`, `rich`) needed by the CLI.

## Quickstart

```bash
UNSTRACT_SRC_PLATFORM_KEY=src_pk_...  \
UNSTRACT_TGT_PLATFORM_KEY=tgt_pk_...  \
uv run python -m unstract.clone clone \
  --source-url https://source.example.com \
  --source-org my-source-org \
  --target-url https://target.example.com \
  --target-org my-target-org
```

Both keys must be **org admin Platform API keys**.

> [!WARNING]
> Both keys grant broad access. Run from a trusted machine and rotate both keys after the clone completes.

> [!NOTE]
> **Unstract Cloud free-trial adapters are not cloned.** Trial adapters are platform-owned and filtered out of the source listing. Prompt Studio projects whose default profile references them are skipped, and that cascades to dependent workflows, API deployments, and pipelines. Provision your own adapters on the target org and re-run the clone to bring the rest across.

> [!NOTE]
> **OAuth-backed connectors need re-authorisation on target.** Connectors that use OAuth (e.g. Google Drive) are cloned without their refresh tokens — the Platform API never exposes them. Re-connect each one on the target after the clone.

## Re-runs are safe

If a phase fails partway, fix the cause and re-run the same command. Resources already on the target are detected by name and reused. There is no `--resume-from` flag — the target is the state.

## Files

The Prompt Studio document corpus is the only resource type with bytes on disk. Default cap per file is 25 MB; oversize files are reported for manual re-upload. Use `--skip-files` to skip bytes entirely (document records are still created).

> [!WARNING]
> Run clones during low-activity windows. Concurrent uploads to the source org during a clone can create duplicate file records on the target.

See the [public docs](https://docs.unstract.com/unstract/unstract_platform/api_documentation/versions/cloning-orgs/) for the full flag list, behavioral notes, and the format of the end-of-run report.
