# Metric Thresholds & Diagnostic Patterns

競馬予測モデルの評価指標閾値とクロスメトリック診断パターン。

## Domain Context

- 典型的なレースは 14-18 頭立て → ランダム的中率 約 5.6-7.1%
- 競馬は本質的に不確実性の高い予測問題（人間の専門家でも的中率 10-15% 程度）
- AUC-ROC 0.50 = ランダム、0.60 以上で意味のある判別力
- 的中精度と収益性は必ずしも相関しない（低オッズ馬の予測は簡単だが回収率が低い）

## Absolute Thresholds

### Accuracy Metrics（的中精度系）

| 指標 | Good | Acceptable | Poor | 根拠 |
|------|------|------------|------|------|
| win_accuracy | >= 0.10 | 0.07-0.10 | < 0.07 | ランダム ~5.6%（18頭）の約2倍が Good 基準 |
| place_accuracy | >= 0.30 | 0.20-0.30 | < 0.20 | Top-1 pick が複勝圏内（3着以内）に入る確率 |
| top3_accuracy | >= 0.35 | 0.25-0.35 | < 0.25 | 予測上位3頭と実際の上位3頭の重複率 |

### ML Performance Metrics（ML性能系）

| 指標 | Good | Acceptable | Poor | 根拠 |
|------|------|------------|------|------|
| auc_roc | >= 0.65 | 0.58-0.65 | < 0.58 | 0.50 = ランダム。0.58 でアラート閾値 (monitor.py: 0.55) を上回る |
| ndcg | >= 0.80 | 0.70-0.80 | < 0.70 | フルランキング品質。0.70 以上で実用的 |
| ndcg_at_3 | >= 0.45 | 0.35-0.45 | < 0.35 | 上位3頭のランキング品質 |
| f1 | >= 0.15 | 0.10-0.15 | < 0.10 | 不均衡分類（勝率 ~6%）のため全体的に低い値が正常 |
| log_loss | <= 0.20 | 0.20-0.25 | > 0.25 | 低いほど良い。確率キャリブレーションの品質指標 |

### Profitability Metrics（収益系）

| 指標 | Good | Acceptable | Poor | 根拠 |
|------|------|------------|------|------|
| recovery_rate | >= 0.85 | 0.70-0.85 | < 0.70 | 1.0 = 損益分岐。控除率 ~25% を考慮 |
| roi | >= -0.10 | -0.25--0.10 | < -0.25 | 0% = 損益分岐。-25% が理論的な控除率相当 |
| expected_value | >= 0.0 | -0.15-0.0 | < -0.15 | 0 以上で期待値プラス |

Note: 収益系指標はベッティングデータ（オッズ）の有無に依存する。データがない場合はスキップする。

## Relationship to Existing Alert Thresholds

`src/evaluator/monitor.py` の既存アラート閾値との関係:

| 指標 | monitor.py 閾値 | 本レビュー Poor | 位置づけ |
|------|-----------------|----------------|----------|
| win_accuracy | 0.05 | < 0.07 | monitor は最低限、本レビューは品質評価 |
| auc_roc | 0.55 | < 0.58 | 同上 |

monitor.py のアラートは「モデルが壊れている」レベル。本レビューの Poor は「改善が必要」レベル。

## Stability Thresholds

バックテスト期間別のメトリクス安定性評価:

| 指標 | 閾値 | 判定 | 意味 |
|------|------|------|------|
| CV (Coefficient of Variation) | > 0.30 | 高分散 | モデルの性能が不安定 |
| Trend (後半平均 / 前半平均) | < 0.90 | 劣化傾向 | 直近のデータに対する汎化性能が低下 |
| Outlier (|value - mean| > 2×std) | 1期間以上 | 外れ値期間あり | 特定の時期に異常な性能を示す |

### Stability Analysis Method

1. バックテスト期間を前半と後半に分割
2. 各半分の平均を比較（後半/前半の比率を算出）
3. 比率 < 0.90 → 劣化傾向と判定
4. CV が大きい場合、トレンドの信頼性は低い（ノイズの可能性）

## Cross-Metric Diagnostic Patterns

指標間の矛盾から問題を診断するパターン:

### Pattern 1: High AUC-ROC + Low Win Accuracy
- **症状**: AUC-ROC >= 0.65 だが win_accuracy < 0.07
- **診断**: 確率キャリブレーション不良。モデルは相対的な順序は正しく判別しているが、1位予測の精度が低い
- **対策**: Platt scaling / isotonic regression による確率補正。または LambdaRank への切り替え

### Pattern 2: Low Log Loss + Low Accuracy
- **症状**: log_loss <= 0.20 だが win_accuracy < 0.07
- **診断**: 過信傾向。モデルは全体的な確率分布は合っているが、上位予測に自信がなく差がつかない
- **対策**: 特徴量追加による判別力強化。特に上位馬を区別する特徴量（近走成績、オッズ等）

### Pattern 3: High NDCG + Low Top3 Accuracy
- **症状**: ndcg >= 0.80 だが top3_accuracy < 0.25
- **診断**: 全体ランキングは良いが、トップ予測が弱い。中位〜下位の順序は正しいが、上位の入れ替えが頻発
- **対策**: top-K 最適化（LambdaRank with truncation level）。特徴量の見直し（上位馬のシグナル強化）

### Pattern 4: Unstable Win Accuracy + Stable AUC-ROC
- **症状**: win_accuracy の CV > 0.30 だが auc_roc の CV < 0.15
- **診断**: モデルの判別力は安定しているが、的中は運の要素が大きい。head-to-head の予測は安定しているが、レースの結果にノイズが多い
- **対策**: これは正常な範囲の可能性あり。win_accuracy は sample variance が大きい指標。place_accuracy と合わせて判断する

### Pattern 5: Degrading Trend in Recent Periods
- **症状**: 後半期間の平均が前半より10%以上低下
- **診断**: コンセプトドリフト。最近の競馬データの傾向が変化している可能性
- **対策**: 学習窓の調整（短くする）。時間重み付き学習。最近のデータに重みを置く
