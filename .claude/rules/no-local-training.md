# No Local Model Training

Do NOT run model training or ML pipeline execution locally. Local machine resources are limited.

- Never run `uv run python -m src.pipeline.orchestrator` or similar pipeline commands locally
- Accuracy/precision validation must be done on Cloud Run via `/preview-report` skill
- Unit tests and lint checks can run locally
- Use `preview-deploy` label + Cloud Run Job for all model evaluation
