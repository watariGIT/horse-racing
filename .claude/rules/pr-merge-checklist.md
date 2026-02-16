# PRマージ前の検証ルール

PRをマージする前に、以下の検証が完了していることを確認する。
**マージはユーザーの明示的な承認を得てから行うこと。Claudeが自動的にマージしてはならない。**

## 必須（全PR）

- CIチェック（test, lint）が全てパス
- docker-build ジョブがパス（preview-deploy.yaml）
- コードレビュー完了（ユーザー承認）

## CI/CD・インフラ変更時

- `preview-deploy` ラベルを付与し、dev環境へのデプロイが成功
- preview-report skillでパイプラインを実行し、PRコメントでレポートを確認

## モデル・特徴量変更時

- preview-report skillでパイプラインを実行し、精度メトリクスに問題がないことを確認
- 精度が大幅に劣化していないことをPRコメントのレポートで確認

## ドキュメント

- CLAUDE.md / README / config の更新が必要な場合は更新済み
