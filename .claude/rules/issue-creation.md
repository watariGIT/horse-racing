# Issue Creation Rule

- Issue title and description must be written in Japanese.
- File discovered improvements and future work items as GitHub Issues
- Do not include out-of-scope work in the current PR
- Apply appropriate labels:
  - `enhancement`: Feature improvements / new features
  - `bug`: Bug fixes
  - `infrastructure`: Infra / CI/CD related
  - `P0` - `P3`: Priority levels (P0 = highest)

## Post-Merge Issue Filing

After merging a PR, file GitHub Issues for items discovered during the work:

- Non-blocking improvements noted during code review
- Technical debt or refactoring candidates found during exploration/implementation
- Related feature improvements that were out of scope for the current PR
- Test coverage gaps identified during development

Trivial style fixes (formatting, naming preferences, etc.) do not require Issue filing.
