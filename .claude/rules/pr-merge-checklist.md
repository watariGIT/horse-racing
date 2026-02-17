# PR Merge Checklist

Verify the following before merging a PR.
**Merging requires explicit user approval. Claude must NEVER auto-merge.**

## Required (All PRs)

- All CI checks (test, lint) pass
- docker-build job passes (preview-deploy.yaml)
- Code review completed (user approved)

## CI/CD / Infrastructure Changes

- Add `preview-deploy` label and confirm dev deployment succeeds
- Run preview-report skill and verify metrics in PR comment

## Model / Feature Changes

- Run preview-report skill and verify accuracy metrics
- Confirm no significant accuracy degradation in PR comment report

## Documentation

- CLAUDE.md / README / config updated if needed
