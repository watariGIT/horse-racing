---
name: address-review-findings
description: Triage and address code review findings from review-pr. Use after /review-pr has posted a review comment, when the user wants to fix WARNING/CRITICAL findings, or says "/address-review-findings". Enters plan mode to classify findings, implements fixes, posts a response comment on the PR, and files GitHub Issues for deferred items.
---

# Address Review Findings

## Prerequisites

- review-pr skill has already posted a `<!-- code-review-report -->` comment on the PR
- `gh` CLI authenticated

## Workflow

1. **Get review comment**

   ```bash
   gh api repos/{owner}/{repo}/issues/{pr_number}/comments \
     --jq '.[] | select(.body | test("code-review-report")) | .body'
   ```

   Parse all findings by severity (CRITICAL / WARNING / SUGGESTION) and domain.

2. **Enter plan mode and classify findings**

   Use `EnterPlanMode` to analyze findings. For each WARNING and CRITICAL:

   - Read the flagged file and line to understand context
   - Check `review-pr/references/review-exceptions.md` for known exceptions
   - Classify into one of three categories:

   | Category | Criteria |
   |----------|----------|
   | **A. Fix** | Real issue, fixable in this PR |
   | **B. Dismiss** | False positive, intentional design, or already addressed |
   | **C. Issue** | Valid but out of scope for this PR |

   Write the classification plan and call `ExitPlanMode` for user approval.

3. **Implement fixes (Category A)**

   - Apply code changes for each "Fix" item
   - Run tests and lint to verify: `uv run pytest tests/ -v`, `uv run ruff check src/ tests/`, `uv run black --check src/ tests/`, `uv run mypy src/`
   - Commit and push

4. **Update review-exceptions.md (Category B)**

   For dismissed findings, append to `.claude/skills/review-pr/references/review-exceptions.md` so they are not re-flagged in future reviews. Commit together with fixes.

5. **Create GitHub Issues (Category C)**

   ```bash
   gh issue create --title "{title}" --body "{body}" --label "infrastructure,P2"
   ```

   Use Japanese for title and body. Apply appropriate labels.

6. **Post PR response comment**

   Post a comment on the PR summarizing all actions taken:

   ```bash
   gh api repos/{owner}/{repo}/issues/{pr_number}/comments -f body="..."
   ```

   Format:

   ```markdown
   ## レビュー指摘対応結果

   **対応コミット**: `{short_sha}`

   ### A. 修正済み（{n}件）

   | 指摘 | 対応 |
   |------|------|
   | {finding description} | {what was done} |

   ### B. 対応不要（却下）（{n}件）

   | 指摘 | 理由 |
   |------|------|
   | {finding description} | {dismissal reason} |

   ### C. Issue 作成（{n}件）

   | Issue | 内容 |
   |-------|------|
   | #{number} | {issue title} |
   ```

## Notes

- SUGGESTION findings are optional. Include in plan if user explicitly requests.
- All text (PR comments, Issues) must be in Japanese.
- Dismissed findings must always be recorded in `review-exceptions.md` to prevent re-flagging.
