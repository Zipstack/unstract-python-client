---
name: spec-upgrade
description: Sync the OpenAPI spec from the Unstract backend into this repo and regenerate the transport layer. Use whenever the backend adds or changes a Document Studio API deployment endpoint, when the `sdk-drift` CI job goes red, when `specs/docstudio-oss.json` is behind the backend, or when someone asks to "regenerate the SDK", "refresh the spec", "pick up the new endpoint", or "release unstract-client". Reach for this even when the request only mentions the generated code or a new endpoint — the upgrade is an ordered pipeline, and skipping a step leaves the drift gate red or the spec silently stale.
---

# Upgrading the spec and regenerating the client

Nothing tells this repo when the backend's spec moves. Someone notices, and then
runs this pipeline. The steps are ordered because each one's output is the next
one's input, and because two of them are the only thing standing between a stale
copy and a client that looks generated but isn't.

## The pieces

| Thing | Where |
|---|---|
| Vendored spec | `specs/docstudio-oss.json` |
| Generator wrapper | `tools/gen_sdk.sh` (pins the generator, records `SPEC_SOURCE`) |
| Generated transport | `src/unstract/api_deployments/_sdk_docstudio/` — never hand-edited |
| Hand-written facade | `src/unstract/api_deployments/client.py` |
| Drift gate | `sdk-drift` job in `.github/workflows/test.yml` |
| Release | `.github/workflows/main.yml`, `workflow_dispatch` |

Upstream, the spec is produced by the backend that serves these endpoints
(`Zipstack/unstract`, `manage.py generate_docstudio_spec`, committed at
`specs/docstudio-oss.json`). It is a copy here, not a fork.

## The sequence

1. **Copy the spec byte-for-byte** from the backend commit you are upgrading to.
   Do not reformat it and do not edit it here — the drift gate regenerates from
   whatever is committed, so an edit here becomes a client that no backend
   serves.

2. **Move `SPEC_SOURCE` in `tools/gen_sdk.sh` in the same commit.** It records
   the backend repo, path, revision and sha256. Without that update, nothing
   distinguishes a current copy from one the backend has moved past — the script
   and the drift gate both report clean either way, which is exactly the failure
   this pipeline exists to prevent.

3. **Regenerate:**

   ```bash
   ./tools/gen_sdk.sh
   ```

   It rebuilds the tree from scratch with a pinned generator. If it exits
   non-zero on a generator warning, believe it: the generator downgrades a
   schema it cannot parse to a warning, drops the endpoint or model, writes the
   rest and exits 0 — so the warning is the only signal that an operation is
   missing.

4. **Review the generated surface, with new files included:**

   ```bash
   git add -N -- src/unstract/api_deployments/_sdk_docstudio
   git diff --stat -- src/unstract/api_deployments/_sdk_docstudio
   ```

   `git add -N` first because a plain diff cannot see a file the generator has
   newly created — which is precisely what a spec that grew an endpoint
   produces. Read the diff as the API change it represents: new modules are new
   operations, removed fields are removals.

5. **Make the new surface reachable, in `client.py`.** Regeneration only moves
   the transport. If the spec added an endpoint callers should be able to use,
   the facade is where it becomes public API. Fixes belong here or upstream in
   the spec, never in the generated tree — regeneration overwrites that wholesale.

6. **Run the tests:** `uv run pytest tests/`. `tests/test_compat.py` compares
   this client against the last released one, vendored under `tests/baseline/`.
   Refresh that baseline only when you mean to move the parity reference point,
   with `tools/refresh_baseline.sh <released-version>`; a spec upgrade on its own
   is not a reason to move it.

## CI

`sdk-drift` regenerates from the committed spec and fails on any diff, so a
hand-edit of the generated tree and a spec change nobody regenerated over both
fail the same way. If it is red, run step 3 and commit the result.

## Versioning and release

Choose the bump by what changed for callers: **minor** for new endpoints or new
behaviour, **patch** for fixes that keep the surface identical.

Do not touch `__version__` in `src/unstract/api_deployments/__init__.py` in your
PR. The in-repo value is the *last released* version; `main.yml` reads it,
applies the bump you pick at dispatch time, and commits the result itself. Bumping
it in the PR makes the release skip a version.

Release by dispatching **Release Tag and Publish Package** on `main` and choosing
the bump. It publishes to PyPI *before* it tags — a failure after publish is
retried by hand against a live artifact, not by re-publishing.

## Downstream

`unstract-cli` pins this client exactly and vendors a copy of the same spec.
After a release lands on PyPI, that pin needs bumping there; see the
`bump-client-pins` skill in that repo.
