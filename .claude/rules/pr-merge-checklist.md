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

## Post-PR Workflow (All PRs)

After creating a PR, execute the following skills in order:

1. Run `/review-pr` to receive a code review
2. Run `/address-review-findings` to address review findings

## Post-PR Workflow (Model / Feature Changes)

After completing the "All PRs" workflow above:

1. Confirm with the user that the system is working correctly
2. Run `/run-pipeline-dev` to validate accuracy metrics
3. Run `/review-accuracy` to receive a model review

## CI/CD / Infrastructure Changes

- Add `preview-deploy` label and confirm dev deployment succeeds
- Run run-pipeline-dev skill and verify metrics in PR comment

## Model / Feature Changes

- Run run-pipeline-dev skill and verify accuracy metrics
- Confirm no significant accuracy degradation in PR comment report

## Documentation

- CLAUDE.md / README / config updated if needed
