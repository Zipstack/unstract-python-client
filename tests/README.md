# Tests

## Unit Tests

Mocked tests that require no external setup:

```bash
uv run pytest -s -v tests/
```

## Integration Test (`client_test.py`)

This test runs against a live Unstract API deployment.

### Setup

1. Copy `tests/sample.env` to `.env` in the **project root**:
   ```bash
   cp tests/sample.env .env
   ```
2. Fill in the values:
   - `API_URL` — your API deployment URL
   - `UNSTRACT_API_DEPLOYMENT_KEY` — your raw API key (**without** the `"Bearer "` prefix; the client adds it automatically)
   - `TEST_FILES` — comma-separated paths to files for structuring (e.g. `/path/to/test1.pdf,/path/to/test2.pdf`)

### Run

```bash
uv run python tests/client_test.py
```
