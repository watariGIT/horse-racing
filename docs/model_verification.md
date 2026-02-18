# Model Verification Guide

## Pipeline Execution

### Local Execution (with Kaggle CSV)

```bash
# 1. Download dataset
kaggle datasets download -d takamotoki/jra-horse-racing-dataset -p data/raw/kaggle/ --unzip

# 2. Run full pipeline
uv run python -m src.pipeline --stage full --date-from 2015-01-01 --date-to 2021-07-31

# 3. Run specific stages
uv run python -m src.pipeline --stage import --date-from 2020-01-01 --date-to 2021-07-31
uv run python -m src.pipeline --stage train --date-from 2015-01-01 --date-to 2021-07-31
```

### Cloud Run Jobs Execution

```bash
# Execute the deployed job
gcloud run jobs execute ml-pipeline \
  --region us-central1 \
  --project horse-racing-ml-dev \
  --wait

# Check execution logs
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=ml-pipeline" \
  --project horse-racing-ml-dev --limit 100

# Retrieve results via MLflow
gcloud run services proxy mlflow-ui-dev --region us-central1 --project horse-racing-ml-dev --port 5000 &
PROXY_PID=$!
sleep 3
uv run python -m src.model_training.compare_experiments --last 1
kill $PROXY_PID 2>/dev/null
```

## Evaluation Metrics

### Accuracy Metrics

| Metric | Description | Baseline | Good |
|--------|-------------|----------|------|
| Win Accuracy | Fraction of races where top pick wins | ~6-8% (random = 1/N) | >10% |
| Place Accuracy | Top pick finishes in top 3 | ~20-25% | >30% |
| Top-3 Accuracy | Overlap of predicted top 3 with actual top 3 | ~25-30% | >35% |

### ML Metrics

| Metric | Description | Expected Range |
|--------|-------------|----------------|
| AUC-ROC | Discrimination ability | 0.55-0.75 |
| Log Loss | Calibration quality | 0.20-0.40 |
| Precision | Among predicted wins, actual wins | 0.05-0.20 |
| Recall | Among actual wins, predicted wins | 0.10-0.50 |
| NDCG | Ranking quality | 0.60-0.85 |

### Interpreting Results

- **AUC-ROC > 0.60**: Model has meaningful signal beyond random guessing
- **Win Accuracy > 1/avg_entries**: Model beats uniform random selection
- **NDCG > 0.70**: Model ranks horses reasonably well within races
- **Train AUC >> Val AUC** (gap > 0.10): Possible overfitting

## Backtest Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train-window` | 365 | Days of training data per backtest period |
| `--test-window` | 30 | Days of test data per backtest period |
| `--date-from` | None | Start of data range |
| `--date-to` | None | End of data range |

### Recommended Configurations

**Quick validation** (fast, fewer periods):
```bash
uv run python -m src.pipeline --stage full \
  --date-from 2020-01-01 --date-to 2021-07-31 \
  --train-window 180 --test-window 30
```

**Comprehensive backtest** (slow, many periods):
```bash
uv run python -m src.pipeline --stage full \
  --date-from 2015-01-01 --date-to 2021-07-31 \
  --train-window 365 --test-window 30
```

## Overfitting Checks

1. **Train/Val gap**: Compare `train_accuracy` vs `val_accuracy` in training logs. Gap > 10% suggests overfitting.
2. **Temporal degradation**: In backtest report, check if later periods perform worse than earlier ones. Consistent degradation suggests concept drift.
3. **Feature importances**: If a single feature dominates (>50% importance), the model may be fragile.
4. **Cross-validation stability**: Check `cv_std_*` metrics. High standard deviation (>0.05 for AUC) suggests unstable training.

## Report Structure

The backtest report includes:

1. **Overall Metrics**: Aggregated performance across all test periods
2. **Period-wise Results**: Table showing metrics for each backtest window

Key columns in period table:
- **Period**: Sequential index
- **Test Range**: Date range of the test window
- **N Test**: Number of test samples
- **Win Accuracy / Place Accuracy / AUC-ROC**: Core metrics per period

## Improvement Cycle

1. **Baseline**: Run pipeline with default settings, record metrics
2. **Feature analysis**: Review `feature_importances.json` for signal quality
3. **Add features**: Implement new extractors (odds, jockey history, etc.)
4. **Tune hyperparameters**: Use Optuna integration for automated tuning
5. **Model comparison**: Compare classifier vs ranker performance
6. **Re-evaluate**: Run backtest with same date range, compare metrics
