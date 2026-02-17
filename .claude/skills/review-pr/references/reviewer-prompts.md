# レビュードメイン別チェックリスト

各レビュアーは担当ドメインのチェックリストに基づいてPR差分を分析する。
指摘には必ず深刻度（CRITICAL / WARNING / SUGGESTION）、ファイルパス、行番号（特定可能な場合）を付与すること。

---

## Engineering（エンジニアリング）

- **実装の正確性**: ロジックの誤り、エッジケースの未処理、off-by-oneエラー
- **テスト**: 変更に対応するテストの有無、テストカバレッジの妥当性、テストの質
- **可読性**: 変数名・関数名の適切さ、複雑な処理の説明
- **型安全性**: 型ヒントの有無（mypy `disallow_untyped_defs=true`）、型の正確性、`Any`の不適切な使用
- **例外処理**: 適切な例外のキャッチ、エラーメッセージの有用性、例外の握りつぶし
- **コーディング規約**: Black（line-length=88）、Ruff（E,F,I,N,W,UP）、Google-style docstring
- **Conventional Commits**: コミットメッセージが `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:` に従っているか

### 確認すべきドキュメント
- `CLAUDE.md`（Coding Standards セクション）
- `pyproject.toml`（Ruff/mypy設定）

---

## Security（セキュリティ）

- **シークレット漏洩**: ハードコードされたAPI鍵、パスワード、トークン、認証情報
- **IAM最小権限**: GCPサービスアカウントの権限が必要最小限か、過剰な権限付与
- **依存関係の脆弱性**: 新しい依存関係の追加、バージョン固定の有無
- **ログ漏洩**: 機密情報（APIキー、個人情報）がログに出力されていないか（structlog使用）
- **入力バリデーション**: 外部入力の検証、インジェクション対策
- **.envファイル管理**: `.gitignore`に含まれているか、`.env.example`との整合性
- **Workload Identity Federation**: GitHub Actions認証がOIDCベースか、長期間トークンの不使用

### 確認すべきドキュメント
- `.env.example`
- `infrastructure/terraform/` 内のIAM関連リソース
- `.gitignore`

---

## Architecture（アーキテクチャ）

- **依存関係の方向**: モジュール間の依存が適切な方向か（`common` ← 他モジュール）
- **境界の明確さ**: モジュール間のインターフェースが明確か、責務が適切に分離されているか
- **冪等性**: バッチ処理（Cloud Run Jobs）が再実行可能か、部分的失敗からの回復
- **データフロー**: GCS → BigQuery → GCS のデータフローが一貫しているか
- **可観測性**: structlogによるログ出力が適切か、エラー追跡が可能か
- **設定管理**: Pydantic Settings + YAML の優先順位（defaults → base.yaml → env.yaml → env vars）に従っているか
- **単一責任**: 各モジュール・クラスが単一の責任を持っているか

### 確認すべきドキュメント
- `CLAUDE.md`（Architecture / Module Structure セクション）
- `config/base.yaml`, `config/dev.yaml`, `config/prod.yaml`

---

## Infrastructure（インフラ）

- **Terraform設計**: リソース定義の適切さ、Workspace（dev/prod）の使い分け、state管理
- **GCP設計**: サービス選択の妥当性、Free Tier活用
- **コスト影響**: 新しいリソース追加によるコスト増加の有無（目標: <$1/月）
- **SLO/リトライ**: Cloud Run Jobsのタイムアウト設定、リトライポリシー、tenacityの使用
- **デプロイ**: GitHub Actions CI/CDの設定、Artifact Registryのクリーンアップポリシー（最新5件保持）
- **ライフサイクル**: GCSライフサイクルルール（90日→Nearline移行）
- **環境分離**: dev/prod環境の適切な分離、環境変数の使い分け

### 確認すべきドキュメント
- `infrastructure/terraform/` 内のリソース定義
- `.github/workflows/` 内のCI/CDワークフロー
- `.claude/rules/cost-management.md`

---

## Data/ML（データ・機械学習）

- **時間的データ漏洩**: 未来の情報が訓練データに混入していないか、特徴量計算で未来データを参照していないか
- **データ分割**: 訓練/検証/テストの分割が時系列順か、ランダム分割による漏洩がないか
- **特徴量の一貫性**: 訓練時と予測時で同じ特徴量パイプラインが使用されているか
- **再現性**: シード値の固定、MLflow実験追跡の設定、`mlflow_run_id`の記録
- **評価設計**: バックテストの方法論が適切か、評価指標（Win/Place/Top3 Accuracy, NDCG, AUC, F1）の妥当性
- **特徴量エンジニアリング**: Polarsを使用した効率的な処理、不要な特徴量の混入
- **モデル管理**: LightGBMモデルのGCS保存、MLflowメタデータ連携、モデルバージョニング

### 確認すべきドキュメント
- `CLAUDE.md`（MLflow Settings セクション）
- `config/base.yaml`（model / feature_engineering セクション）
- `src/feature_engineering/`, `src/model_training/`, `src/evaluator/`
