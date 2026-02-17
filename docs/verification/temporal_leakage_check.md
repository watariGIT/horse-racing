# 時系列データ漏洩検証レポート

## 検証概要

- **目的**: パイプライン初回実行結果に基づく時系列データ漏洩の体系的検証
- **実施日**: 2026-02-17
- **対象**: Issue #8
- **パイプライン初回実行日**: 2026-02-16

## 検証結果サマリ

| 検証項目 | 結果 | 根拠 |
|----------|------|------|
| DataPreparer shift(1) | Pass | shift(1) により現在行を除外、初回出走馬の num_past_races=0 をテストで確認 |
| FeaturePipeline fit/transform 分離 | Pass | Encoder/Scaler は訓練データのみで fit、未学習状態での transform は RuntimeError |
| BacktestEngine temporal split | Pass | train_end < test_start の厳密な時系列分離をテストで確認 |
| Train/Val AUC 差 | Pass | Gap = 0.053 (< 0.1 の許容範囲内) |

## 1. DataPreparer の保護機構

### 仕組み

`src/pipeline/data_preparer.py` の `DataPreparer` クラスは、各レース行に対して過去の履歴統計を計算する。未来データの漏洩を防ぐため、以下の2つの手法を採用している。

**shift(1) による現在行の除外** (92-123行目):

全てのローリング統計計算において `shift(1)` を適用し、現在のレース結果を集計対象から除外する。

```python
# avg_finish: 過去nレースの着順平均
pl.col("finish_position")
    .shift(1)
    .rolling_mean(window_size=n, min_samples=1)
    .over("horse_id")
    .alias("avg_finish"),
```

この処理は `win_rate` (100-105行目)、`top3_rate` (107-112行目) でも同様に適用される。

**cum_count - 1 による過去出走数の計算** (118-122行目):

`num_past_races` は累積カウントから1を引くことで、現在のレースを除外した過去出走数を算出する。

```python
pl.col("finish_position")
    .cum_count()
    .over("horse_id")
    .sub(1)
    .alias("num_past_races"),
```

初出走の馬は `num_past_races=0` となり、参照可能な過去データが存在しないことを正しく表現する。

### 検証結果

以下の既存テストにより、データ漏洩防止が確認されている。

**test_no_future_data_leakage** (`tests/unit/test_pipeline.py:60-70`):
各馬の初レースにおいて `num_past_races == 0` であることを検証。shift(1) により初回出走時に未来データが参照されないことを保証する。

**test_rolling_stats_accumulate** (`tests/unit/test_pipeline.py:120-132`):
各馬の `num_past_races` がレースごとに単調増加することを検証。過去データのみが累積されていることを確認する。

## 2. FeaturePipeline の保護機構

### 仕組み

`src/feature_engineering/pipeline.py` の `FeaturePipeline` クラスは、scikit-learn 互換の fit/transform API を提供し、訓練データとテストデータの分離を保証する。

**ステートレスな Feature Extractor** (206-218行目):

`RaceFeatureExtractor`、`HorseFeatureExtractor`、`JockeyFeatureExtractor` はすべてステートレスであり、入力データのカラム変換のみを行う。内部状態を持たないため、データ漏洩のリスクがない。

**fit/transform 分離と RuntimeError ガード** (154-171行目):

`transform()` メソッドは、Encoder または Scaler が設定されている場合、`fit()` が事前に呼び出されていないと `RuntimeError` を発生させる。

```python
if (self._encoder or self._scaler) and not self._fitted:
    raise RuntimeError(
        "Pipeline must be fitted before transform "
        "when encoder or scaler is configured"
    )
```

**Encoder/Scaler の訓練データ限定学習** (118-152行目):

`fit()` メソッドでは、Encoder と Scaler を訓練データのみで学習する。`transform()` 時にはこの学習済みパラメータを使用するため、テストデータの統計情報が学習に混入することはない。

### 検証結果

以下の新規テストにより、fit/transform 境界の正当性が確認されている。

**TestFeaturePipelineFitTransformBoundary** (`tests/unit/test_feature_engineering.py:504-574`):

- **test_encoder_fitted_on_train_only** (507-525行目): Encoder のマッピングが訓練データのカテゴリのみを含むことを検証。テストデータに存在する未知カテゴリ ("Hanshin") は -1 にマッピングされる。
- **test_scaler_fitted_on_train_only** (527-541行目): Scaler の統計値 (mean, std) が訓練データの分布のみを反映していることを検証。
- **test_transform_before_fit_raises** (543-553行目): 未学習状態での `transform()` 呼び出しが `RuntimeError` を発生させることを検証。
- **test_transform_uses_fitted_stats_not_new_data** (555-574行目): テストデータの変換に訓練データの統計値が使用されることを検証。テストデータ独自の分布は無視される。

## 3. BacktestEngine の保護機構

### 仕組み

`src/evaluator/backtest_engine.py` の `BacktestEngine` クラスは、Walk-forward バックテストにおいて厳密な時系列分離を実施する。

**時系列分離によるデータ分割** (164-171行目):

訓練データとテストデータの分割は日付ベースで行われ、訓練データは `date < test_start` の条件で厳密にフィルタリングされる。

```python
train_df = df.filter(
    (pl.col(self._date_col).cast(pl.Date) >= train_start)
    & (pl.col(self._date_col).cast(pl.Date) < test_start)
)
test_df = df.filter(
    (pl.col(self._date_col).cast(pl.Date) >= test_start)
    & (pl.col(self._date_col).cast(pl.Date) <= test_end)
)
```

`train_df` の条件は `< test_start` (厳密な未満)、`test_df` の条件は `>= test_start` であるため、訓練期間とテスト期間にデータの重複は発生しない。

### 検証結果

以下の新規テストにより、時系列分離の正当性が確認されている。

**TestBacktestTemporalIntegrity.test_no_temporal_overlap** (`tests/unit/test_evaluator.py:770-817`):

150日分のデータでバックテストを実行し、全ての期間において `train_end < test_start` であることを検証。訓練期間とテスト期間が厳密に分離されていることを保証する。

```python
for period in result.periods:
    train_end = date.fromisoformat(period.train_end)
    test_start = date.fromisoformat(period.test_start)
    assert train_end < test_start, (
        f"Period {period.period_index}: train_end ({train_end}) "
        f"must be strictly before test_start ({test_start})"
    )
```

## 4. メトリクス評価

パイプライン初回実行 (2026-02-16) の結果:

| メトリクス | 値 |
|-----------|-----|
| Train AUC | 0.775 |
| Val AUC | 0.722 |
| Gap | 0.053 |

- Train AUC と Val AUC の差は 0.053 であり、許容範囲 (< 0.1) 内に収まっている。
- 差が 0.1 を超える場合は過学習または時系列漏洩の兆候となるが、現在の値はモデルが適切に汎化していることを示す。
- バックテスト期間全体を通じて、異常な AUC スパイクは観測されなかった。

## 5. 結論

以下の4つの観点から検証を実施した結果、重大な時系列データ漏洩は検出されなかった。

1. **DataPreparer**: `shift(1)` と `cum_count().sub(1)` により、各レース行の特徴量計算に現在および未来のデータが混入しないことを確認。既存テスト (`test_no_future_data_leakage`, `test_rolling_stats_accumulate`) で検証済み。

2. **FeaturePipeline**: Encoder/Scaler の fit/transform 分離により、テストデータの統計情報が訓練プロセスに漏洩しないことを確認。新規テスト (`TestFeaturePipelineFitTransformBoundary`) で4項目を検証済み。

3. **BacktestEngine**: `date < test_start` による厳密な時系列分離により、訓練データとテストデータの重複がないことを確認。新規テスト (`TestBacktestTemporalIntegrity.test_no_temporal_overlap`) で検証済み。

4. **メトリクス**: Train AUC (0.775) と Val AUC (0.722) の差 (0.053) は許容範囲内であり、過学習や時系列漏洩の兆候は見られない。
