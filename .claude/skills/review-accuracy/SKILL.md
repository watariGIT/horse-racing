---
name: review-accuracy
description: Review model accuracy results from MLflow and suggest improvements. Use after /run-pipeline-dev has completed, when the user wants to evaluate model performance, check backtest metrics, or generate improvement ideas. Triggered by "/review-accuracy". Posts a PR comment with MLflow experiment link, metric analysis, and files GitHub Issues for improvement ideas.
---

# Review Accuracy

## Prerequisites

- PR exists on current branch
- `/run-pipeline-dev` has been executed and pipeline completed successfully
- `gcloud` CLI authenticated
- `gh` CLI authenticated

## Workflow

1. **Get PR info and MLflow UI URL**

   ```bash
   gh pr view --json number,headRefOid,title
   ```

   ```bash
   gcloud run services describe mlflow-ui-dev \
     --region us-central1 --project horse-racing-ml-dev \
     --format 'value(status.url)'
   ```/

   Store the PR number, title, commit SHA, and MLflow UI base URL.

2. **Fetch experiment data from MLflow and GCS**

   Get the latest model version and its MLflow run ID:

   ```bash
   # List model versions (sorted chronologically by date prefix)
   gsutil ls gs://horse-racing-ml-dev-models/models/win_classifier/
   ```

   Take the latest version directory, then fetch metadata:

   ```bash
   gsutil cat gs://horse-racing-ml-dev-models/models/win_classifier/{latest_version}/metadata.json
   ```

   From metadata, extract:
   - `mlflow_run_id` — for constructing MLflow UI link
   - `metrics` — training metrics
   - `model_type`, `params` — model configuration

   Fetch backtest metrics from MLflow:

   ```bash
   gcloud run services proxy mlflow-ui-dev --region us-central1 --project horse-racing-ml-dev --port 5000 &
   PROXY_PID=$!
   sleep 3
   ```

   ```bash
   uv run python -c "
   import mlflow, json
   mlflow.set_tracking_uri('http://localhost:5000')
   runs = mlflow.search_runs(
       experiment_names=['horse-racing-prediction'],
       max_results=1,
       order_by=['start_time DESC'],
   )
   if not runs.empty:
       row = runs.iloc[0]
       # Overall metrics
       metrics = {}
       for c in sorted(runs.columns):
           if c.startswith('metrics.backtest_overall_'):
               name = c.replace('metrics.backtest_overall_', '')
               metrics[name] = round(row[c], 4)
       print(json.dumps(metrics, indent=2))
       # Download per-period backtest artifact
       run_id = row['run_id']
       local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='backtest_results.json')
       with open(local_path) as f:
           backtest = json.load(f)
       print('---PERIODS---')
       print(json.dumps(backtest.get('periods', []), indent=2))
   "
   ```

   ```bash
   kill $PROXY_PID 2>/dev/null
   ```

   **トラブルシューティング**: proxy 接続エラー時は `lsof -i :5000` でポート競合を確認し、別ポート (`--port 5001`) で再試行する。

   Extract from the output:
   - Overall metrics (JSON before `---PERIODS---`)
   - Per-period results (JSON after `---PERIODS---`)

   Fetch feature importances artifact:

   ```bash
   # Try to fetch from MLflow artifact store
   gsutil cat gs://horse-racing-ml-dev-models/mlartifacts/*//{mlflow_run_id}/artifacts/feature_importances.json
   ```

   If feature importances are not accessible, note it in the report and skip Step 4.

   Construct MLflow UI link:

   ```
   {MLFLOW_UI_URL}/#/experiments/1/runs/{mlflow_run_id}
   ```

   Note: experiment ID は `1` をデフォルトとし、リンクが不正の場合はベースURLのみ掲載する。

3. **Analyze metrics**

   Read `references/metric-thresholds.md` for threshold definitions.

   **A. Absolute Performance Assessment**

   Evaluate each metric against Good / Acceptable / Poor thresholds:

   | 指標 | Good | Acceptable | Poor |
   |------|------|------------|------|
   | win_accuracy | >= 0.10 | 0.07-0.10 | < 0.07 |
   | place_accuracy | >= 0.30 | 0.20-0.30 | < 0.20 |
   | top3_accuracy | >= 0.35 | 0.25-0.35 | < 0.25 |
   | auc_roc | >= 0.65 | 0.58-0.65 | < 0.58 |
   | ndcg | >= 0.80 | 0.70-0.80 | < 0.70 |
   | ndcg_at_3 | >= 0.45 | 0.35-0.45 | < 0.35 |
   | f1 | >= 0.15 | 0.10-0.15 | < 0.10 |
   | log_loss | <= 0.20 | 0.20-0.25 | > 0.25 |

   **B. Period Stability Analysis**

   For each key metric across backtest periods:
   - Mean, std, min, max
   - Coefficient of Variation (CV = std / mean)
   - Trend direction (increasing / stable / degrading based on first-half vs second-half mean)

   Stability flags:
   - CV > 0.3 → 「高分散」
   - Second-half mean < first-half mean × 0.9 → 「劣化傾向」
   - Any period > mean ± 2×std → 「外れ値期間あり」

   **C. Cross-Metric Diagnostic**

   Check for contradictions indicating modeling issues:
   - High AUC-ROC + low win_accuracy → 確率キャリブレーション不良
   - Low log_loss + low accuracy → 過信傾向（overconfident predictions）
   - High NDCG + low top3_accuracy → 全体ランキングは良いがトップ予測が弱い

4. **Analyze feature importance** (if available)

   - Dominant features: any single feature > 30% of total importance → 過度の依存リスク
   - Low-importance features: features < 1% → 削除または再設計候補
   - Group balance: compare total importance across groups (race / horse / jockey / running_style)
   - Identify underutilized feature groups

5. **Determine adoption decision**

   Based on the metric analysis:

   | 判定 | 条件 |
   |------|------|
   | **推奨: 採用** | Poor 指標なし、安定性に重大な問題なし |
   | **条件付き採用** | 一部 Acceptable だが全体として改善または維持。Poor が profitability 系のみ |
   | **要検討** | Accuracy/ML系にPoor指標あり、または安定性に深刻な問題 |

6. **Generate improvement suggestions and create Issues**

   Read `references/improvement-catalog.md` and select applicable suggestions based on:
   - Which metrics are Poor or Acceptable (target those)
   - Feature importance analysis results
   - Cross-metric diagnostic findings

   **Priority assignment:**

   | 優先度 | 基準 | 期待効果 |
   |--------|------|----------|
   | P0 | Poor 指標の修正、モデリング上の欠陥修正 | 主要指標 +5% 以上 |
   | P1 | Acceptable → Good への改善、安定性向上 | 主要指標 +2-5% |
   | P2 | インクリメンタル改善、既存データ活用の新特徴量 | 主要指標 +1-2% |
   | P3 | 実験的アイデア、効果不明 | PR コメントのみ記載 |

   **重複チェック（Issue 作成前に必須）:**

   P0-P2 の各提案について、Issue を作成する前に既存 Issue との重複を確認する:

   ```bash
   gh issue list --state open --label "model-improvement" --json number,title --jq '.[] | "\(.number)\t\(.title)"'
   ```

   各提案のタイトル・内容と既存 Issue を比較し:
   - **重複あり**: Issue を作成せず、既存 Issue の番号を PR コメントの改善提案テーブルに記載する
   - **重複なし**: 新規 Issue を作成する

   判定基準: タイトルが完全一致でなくても、同じ改善目的・対象指標をカバーしている場合は重複とみなす。

   Create GitHub Issues for P0-P2 items only (重複がないもの):

   ```bash
   gh issue create \
     --title "{日本語タイトル}" \
     --body "$(cat <<'EOF'
   ## 概要
   {改善内容の説明}

   ## 背景
   {対象指標と現在の値、問題点}

   ## 期待される効果
   {改善見込みの具体的な指標と幅}

   ## 実装アプローチ
   {高レベルの実装ステップ}

   ## 優先度根拠
   P{n}: {この優先度にした理由}

   ---
   *この Issue は `/review-accuracy` スキルにより自動生成されました*
   EOF
   )" \
     --label "model-improvement,P{n}"
   ```

   P3 items are listed in the PR comment only (no Issue created, to avoid noise).

7. **Post/update PR comment**

   Use the idempotent comment pattern with marker `<!-- accuracy-review-report -->`.

   Search for existing comment:
   ```bash
   gh api repos/{owner}/{repo}/issues/{pr_number}/comments \
     --jq '[.[] | select(.body | test("accuracy-review-report")) | .id] | first'
   ```

   If found: PATCH to update:
   ```bash
   gh api repos/{owner}/{repo}/issues/comments/{comment_id} -X PATCH -f body="..."
   ```

   If not found: POST to create:
   ```bash
   gh api repos/{owner}/{repo}/issues/{pr_number}/comments -f body="..."
   ```

## PR Comment Format

```markdown
<!-- accuracy-review-report -->
## モデル精度レビュー

**Commit**: `{short_sha}`
**MLflow 実験結果**: [{run_name}]({MLFLOW_UI_URL}/#/experiments/{exp_id}/runs/{run_id})

### 採用判定

**{推奨: 採用 / 条件付き採用 / 要検討}**

{判定理由 1-2文}

### 指標サマリー

| カテゴリ | 指標 | 値 | 評価 |
|----------|------|-----|------|
| 的中精度 | Win Accuracy | {value} | {Good/Acceptable/Poor} |
| 的中精度 | Place Accuracy | {value} | {Good/Acceptable/Poor} |
| 的中精度 | Top3 Accuracy | {value} | {Good/Acceptable/Poor} |
| ML性能 | AUC-ROC | {value} | {Good/Acceptable/Poor} |
| ML性能 | NDCG | {value} | {Good/Acceptable/Poor} |
| ML性能 | NDCG@3 | {value} | {Good/Acceptable/Poor} |
| ML性能 | F1 | {value} | {Good/Acceptable/Poor} |
| ML性能 | Log Loss | {value} | {Good/Acceptable/Poor} |

{Cross-metric diagnostic findings があれば記載}

<details>
<summary>期間別安定性分析</summary>

| 指標 | 平均 | 標準偏差 | CV | 傾向 | 判定 |
|------|------|----------|-----|------|------|
| Win Accuracy | {mean} | {std} | {cv} | {trend} | {安定/高分散/劣化傾向} |
| AUC-ROC | {mean} | {std} | {cv} | {trend} | {安定/高分散/劣化傾向} |
| ... | ... | ... | ... | ... | ... |

{外れ値期間がある場合はここで報告}

</details>

<details>
<summary>特徴量重要度分析</summary>

| 特徴量 | 重要度 | グループ | 備考 |
|--------|--------|----------|------|
| {feature} | {importance} | {group} | {支配的/低重要度 etc.} |
| ... | ... | ... | ... |

**グループ別重要度合計**: race: {x}%, horse: {y}%, jockey: {z}%, running_style: {w}%

{所見: 偏りや改善ポイント}

{特徴量データ未取得の場合: "MLflow アーティファクトにアクセスできないためスキップ"}

</details>

### 改善提案

| # | 内容 | 対象指標 | 期待効果 | 優先度 | Issue |
|---|------|----------|----------|--------|-------|
| 1 | {description} | {metrics} | {expected lift} | P{n} | #{issue_number} |
| 2 | {description} | {metrics} | {expected lift} | P3 | -- |

---
*このレビューは `/review-accuracy` スキルによる自動分析です。*
```

## Notes

- すべてのPRコメント・Issueは日本語で記載する
- 再実行時はHTMLマーカーにより既存コメントが更新され、重複しない
- P3アイテムはIssue作成せず、PRコメントにのみ記載する
- `model-improvement` ラベルがリポジトリに存在しない場合、Issue作成時に自動作成される
- 特徴量重要度が取得できない場合はスキップし、レポートにその旨を記載する
- MLflow UIリンクが正しく動作するかは保証しない（Cloud Runのスケーリング状態による）。リンクテキストとしてrun名を表示する
