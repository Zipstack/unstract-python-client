# Release notes — draft

Content for the next release. Not published yet.

## Breaking

**The `unstract` console script is gone.** Installing this package no longer
puts an `unstract` command on your PATH.

- The clone command is still here: run it as `python -m unstract.clone`, with
  the same options it has always taken.
- The name now belongs to the `unstract-cli` package, whose `unstract clone`
  wraps this same code.

An environment that holds both an older release of this package and the new CLI
gives the name to whichever was installed last, so check what answers before
reporting a missing command:

```bash
command -v unstract && unstract --version
```

## Behaviour that differs from earlier releases

Deliberate; each is described in the README under *Behaviour that differs from
earlier releases*:

- The status poll resolves under the deployment URL's own path prefix, and falls
  back to the endpoint the service returned where no prefix can be derived.
- An absolute `status_check_api_endpoint` resolves instead of being concatenated
  into an unreachable URL.
- Query parameters this client sets win a collision with the returned
  endpoint's; everything else on that endpoint is forwarded.
- A malformed `api_url` raises this client's own exception classes rather than
  `InvalidSchema`.

## Under the hood

The HTTP layer is generated from the service's OpenAPI spec and runs on `httpx`.
Transport failures are still raised as their `requests` equivalents, so code
catching `ConnectionError`, `Timeout` and the rest keeps working.
