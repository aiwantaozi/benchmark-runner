# Automated code review

The `Code Review` workflow uses the same
[shared workflow](https://github.com/gpustack/.github/blob/main/.github/workflows/code-review.yml)
as GPUStack. Once `.github/workflows/code-review.yml` is merged into the default
branch, opening, updating, or reopening a pull request triggers an AI review,
including pull requests from forks. Organization members, repository owners, and
collaborators can request another review by starting a PR comment with
`/gpustack-code-review` or `@gpustack-code-review`.

Repository-specific rules live in `.opencodereview/rule.json`. They supplement
OCR's built-in rules for the benchmark runtime, auto-tune search, tests, and README.
The `include` patterns explicitly enable Python tests that OCR otherwise skips;
they do not limit review to only those paths. Dependency locks and build artifacts
listed in `exclude` are skipped.

The `gpustack-code-review` GitHub App must be installed for this repository with
the permissions required by the shared workflow. The following organization-level
Actions variables and secrets must also be accessible to `gpustack/benchmark-runner`:

| Type | Names |
| --- | --- |
| Variables | `CI_GPUSTACK_CODEREVIEW_LLM_URL`, `CI_GPUSTACK_CODEREVIEW_LLM_MODEL`, `CI_GPUSTACK_CODEREVIEW_LLM_THINKING`, `CI_GPUSTACK_CODEREVIEW_LLM_REASONING_EFFORT`, `CI_GPUSTACK_CODEREVIEW_LLM_TEMPERATURE`, `CI_GPUSTACK_CODEREVIEW_LLM_TOP_P`, `CI_GPUSTACK_CODEREVIEW_LLM_THINKING_CLEAR` |
| Secrets | `CI_GPUSTACK_CODEREVIEW_LLM_AUTH_TOKEN`, `CI_GPUSTACK_CODEREVIEW_APP_ID`, `CI_GPUSTACK_CODEREVIEW_APP_PRIVATE_KEY` |

If an organization variable or secret is restricted to selected repositories, an
organization administrator must add this repository to its access list. The
workflow uses `pull_request_target` to access these secrets for fork PRs; keep
execution in the shared workflow and do not add steps that execute PR code here.
