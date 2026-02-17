---
name: review-pr
description: Create a team of 5 specialized reviewer agents to review the current PR and post a consolidated review comment in Japanese. Use when the user wants a code review, asks to review a PR, or says "/review-pr". Covers engineering, security, architecture, infrastructure, and data/ML domains.
---

# Review PR

## Prerequisites

- PR exists on current branch
- `gh` CLI authenticated

## Workflow

1. **Get PR info and diff**
   ```bash
   gh pr view --json number,title,headRefOid,baseRefName
   gh pr diff
   gh pr view --json files --jq '.files[].path'
   ```
   Store the PR number, title, commit SHA, and full diff.

2. **Create reviewer team**

   Use `TeamCreate` to create a team named `pr-review`.

3. **Create review tasks and spawn agents**

   Create 5 tasks with `TaskCreate` and spawn 5 `general-purpose` teammates in parallel using the `Task` tool with `team_name: "pr-review"`.

   Each agent receives:
   - The full PR diff
   - The list of changed files
   - Their domain-specific checklist from `references/reviewer-prompts.md`
   - Instructions below

   **Agent instructions (common)**:
   - PR差分を分析し、担当ドメインの観点から問題を指摘する
   - 差分だけでは文脈が不足する場合、変更されたファイルの全体をReadツールで読む
   - 「確認すべきドキュメント」に記載されたファイルも確認し、ドキュメント更新の必要性を指摘する
   - 各指摘に深刻度を付与する:
     - **CRITICAL**: マージ前に必ず修正（セキュリティ脆弱性、データ漏洩、破壊的変更）
     - **WARNING**: 修正すべき（エラーハンドリング不足、潜在バグ、テスト不足）
     - **SUGGESTION**: 改善提案（スタイル改善、軽微なリファクタリング）
   - 出力フォーマット（1指摘につき1行）:
     ```
     - **[CRITICAL]** `path/to/file` (L42) - 説明文（日本語）
     ```
   - 指摘がない場合は「指摘事項なし」と回答する
   - すべて日本語で回答する

   **Agent names**: `engineering`, `security`, `architecture`, `infrastructure`, `data-ml`

4. **Collect results**

   Wait for all 5 agents to complete. Collect findings from each agent via their SendMessage responses.

5. **Aggregate and format comment**

   Combine all findings into the PR Comment Format below:
   - Count findings per severity (CRITICAL / WARNING / SUGGESTION) across all domains
   - Each domain is a collapsible `<details>` section
   - If a domain has no findings, show "指摘事項なし"
   - If CRITICAL count > 0, add warning message

6. **Post/update PR comment** using `gh api`:
   - Search for existing comment with marker `<!-- code-review-report -->`:
     ```bash
     gh api repos/{owner}/{repo}/issues/{pr_number}/comments --jq '.[] | select(.body | contains("<!-- code-review-report -->")) | .id'
     ```
   - If found: PATCH to update
     ```bash
     gh api repos/{owner}/{repo}/issues/comments/{comment_id} -X PATCH -f body="..."
     ```
   - If not found: POST to create
     ```bash
     gh api repos/{owner}/{repo}/issues/{pr_number}/comments -f body="..."
     ```

7. **Shutdown team**

   Send `shutdown_request` to all 5 agents. After confirmation, use `TeamDelete`.

## PR Comment Format

```markdown
<!-- code-review-report -->
## Code Review Report

**PR**: #{number} {title}
**Commit**: `{short_sha}`
**レビュー日時**: {YYYY-MM-DD HH:mm}

### サマリー
| 深刻度 | 件数 |
|--------|------|
| CRITICAL | {n} |
| WARNING | {n} |
| SUGGESTION | {n} |

{CRITICAL > 0 の場合のみ表示: "⚠️ **CRITICAL指摘があります。マージ前に対応が必要です。**"}

<details>
<summary>Engineering ({n}件)</summary>

- **[SEVERITY]** `path/to/file` (L{line}) - {説明}

</details>

<details>
<summary>Security ({n}件)</summary>

{findings or "指摘事項なし"}

</details>

<details>
<summary>Architecture ({n}件)</summary>

{findings or "指摘事項なし"}

</details>

<details>
<summary>Infrastructure ({n}件)</summary>

{findings or "指摘事項なし"}

</details>

<details>
<summary>Data/ML ({n}件)</summary>

{findings or "指摘事項なし"}

</details>

---
*このレビューはClaude Codeによる自動レビューです。*
```

## Notes

- PR差分が非常に大きい場合（500行超）、最もインパクトの大きい変更に集中するようレビュアーに指示する
- 再実行時はHTMLマーカーにより既存コメントが更新され、重複しない
