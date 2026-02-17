# PR Merge Checklist

Verify the following before merging a PR.
**Merging requires explicit user approval. Claude must NEVER auto-merge.**

## Language

- PR title, description, and all comments must be written in Japanese.

## Required (All PRs)

- All CI checks (test, lint) pass
- docker-build job passes (preview-deploy.yaml)
- Code review completed (user approved)
- Run review-pr skill: CRITICAL findings must be fixed before merge; WARNING/SUGGESTION findings should be addressed or filed as GitHub Issues
- After addressing review findings, post a PR comment documenting: what was fixed (with commit SHA), what was deemed unnecessary to fix (with reason), and what was filed as Issues (with Issue numbers)

## CI/CD / Infrastructure Changes

- Add `preview-deploy` label and confirm dev deployment succeeds
- Run preview-report skill and verify metrics in PR comment

## Model / Feature Changes

- Run preview-report skill and verify accuracy metrics
- Confirm no significant accuracy degradation in PR comment report

## Documentation

- CLAUDE.md / README / config updated if needed
