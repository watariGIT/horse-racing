---
name: review-pr
description: Create a team of specialized reviewer agents (dynamically selected based on changed files) to review the current PR and post a consolidated review comment in Japanese. Use when the user wants a code review, asks to review a PR, or says "/review-pr". Covers engineering, security, architecture, infrastructure, and data/ML domains as needed.
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

2. **Determine relevant reviewers based on changed files**

   Analyze the list of changed files from step 1 and select which reviewer domains to activate:

   | ファイルパターン | 対象レビュアー |
   |----------------|--------------|
   | `*.py` | engineering, security, architecture |
   | `src/feature_engineering/**`, `src/model_training/**`, `src/evaluator/**`, `src/predictor/**` | data-ml |
   | `infrastructure/terraform/**`, `.github/workflows/**`, `Dockerfile*`, `infrastructure/mlflow/**` | infrastructure |
   | `config/**` | architecture |
   | `pyproject.toml`, `uv.lock`, `*.yaml`, `*.toml`, `*.json`（config/以外） | engineering |
   | `.claude/**`（スキル定義、ルール等） | engineering |
   | `.md` ファイルのみ（他のファイル種別なし） | engineering のみ |

   ルール:
   - `engineering` は常に含める（ベースラインレビュアー）
   - 各ドメインは、変更ファイルのいずれかがパターンに一致する場合に選択される
   - すべての変更ファイルが `.md` のみの場合は `engineering` のみ選択

   判定ロジック:
   ```
   reviewers = {"engineering"}  # 常に含む

   if any file ends with .py:
       reviewers.add("security")
       reviewers.add("architecture")

   if any file matches infrastructure/terraform/** or .github/workflows/** or Dockerfile* or infrastructure/mlflow/**:
       reviewers.add("infrastructure")

   if any file matches src/feature_engineering/** or src/model_training/** or src/evaluator/** or src/predictor/**:
       reviewers.add("data-ml")

   if any file matches config/**:
       reviewers.add("architecture")

   # pyproject.toml, uv.lock, .claude/** 等はengineeringのみ（既に含まれている）

   # 特殊ケース: .md ファイルのみの場合
   if all changed files end with .md:
       reviewers = {"engineering"}
   ```

3. **Create reviewer team**

   Use `TeamCreate` to create a team named `pr-review`.

4. **Create review tasks and spawn agents**

   Create one task per selected reviewer domain from step 2, and spawn one `general-purpose` teammate per domain using the `Task` tool with `team_name: "pr-review"`.

   Each agent receives:
   - The full PR diff
   - The list of changed files
   - Their domain-specific checklist from `references/reviewer-prompts.md`
   - Instructions below

   **Agent instructions (common)**:
   - PR差分を分析し、担当ドメインの観点から問題を指摘する
   - 差分だけでは文脈が不足する場合、変更されたファイルの全体をReadツールで読む
   - 「確認すべきドキュメント」に記載されたファイルも確認し、ドキュメント更新の必要性を指摘する
   - `references/review-exceptions.md` に記載された既知の設計判断は指摘対象外とする
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

   **Agent names**: Use the domain names selected in step 2 (subset of: `engineering`, `security`, `architecture`, `infrastructure`, `data-ml`)

5. **Collect results**

   Wait for all spawned agents to complete. Collect findings from each agent via their SendMessage responses.

6. **Aggregate and format comment**

   Combine all findings into the PR Comment Format below:
   - Count findings per severity (CRITICAL / WARNING / SUGGESTION) across all domains
   - Each domain is a collapsible `<details>` section
   - If a domain has no findings, show "指摘事項なし"
   - For domains that were NOT selected in step 2, display「対象外（変更ファイルに該当なし）」instead of findings
   - If CRITICAL count > 0, add warning message

7. **Post/update PR comment** using `gh api`:
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

8. **Shutdown team**

   Send `shutdown_request` to all spawned agents. After confirmation, use `TeamDelete`.

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

{CRITICAL > 0 の場合のみ表示: "**CRITICAL指摘があります。マージ前に対応が必要です。**"}

<details>
<summary>Engineering ({n}件)</summary>

- **[SEVERITY]** `path/to/file` (L{line}) - {説明}

</details>

<details>
<summary>Security ({n}件)</summary>

{findings or "指摘事項なし" or "対象外（変更ファイルに該当なし）"}

</details>

<details>
<summary>Architecture ({n}件)</summary>

{findings or "指摘事項なし" or "対象外（変更ファイルに該当なし）"}

</details>

<details>
<summary>Infrastructure ({n}件)</summary>

{findings or "指摘事項なし" or "対象外（変更ファイルに該当なし）"}

</details>

<details>
<summary>Data/ML ({n}件)</summary>

{findings or "指摘事項なし" or "対象外（変更ファイルに該当なし）"}

</details>

---
*このレビューはClaude Codeによる自動レビューです。*
```

※ ステップ2で選択されなかったドメインは「対象外（変更ファイルに該当なし）」と表示する

## Notes

- PR差分が非常に大きい場合（500行超）、最もインパクトの大きい変更に集中するようレビュアーに指示する
- 再実行時はHTMLマーカーにより既存コメントが更新され、重複しない
