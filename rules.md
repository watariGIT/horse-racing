# rules.md - コーディング規約・レビュー基準

## Python コーディング規約

### スタイル

- PEP 8 準拠
- フォーマッター: **Black** (line-length=88, target-version=py310)
- リンター: **Ruff** (select: E, F, I, N, W, UP)
- 型チェック: **mypy** (disallow_untyped_defs=true)

### 型ヒント

型ヒントは**必須**。すべての関数の引数と戻り値に型を付ける。

```python
# Good
def fetch_race_data(race_id: str, date: datetime.date) -> pd.DataFrame:
    ...

# Bad
def fetch_race_data(race_id, date):
    ...
```

`from __future__ import annotations` を各モジュール先頭でインポートし、`str | None` 形式の Union 構文を使用する。

### Docstring

Google style docstring を使用する。公開関数・クラスには必ず記述する。

```python
def upload_file(
    self, bucket_name: str, source_path: str | Path, destination_blob: str
) -> str:
    """Upload a local file to GCS.

    Args:
        bucket_name: Target GCS bucket name.
        source_path: Path to local file.
        destination_blob: Destination path in the bucket.

    Returns:
        The GCS URI of the uploaded file.

    Raises:
        GoogleCloudError: If the upload fails.
    """
```

### インポート

Ruff の `I` ルールに従い、以下の順序でグループ化する:

1. 標準ライブラリ
2. サードパーティ
3. ローカル（`src.`）

```python
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from google.cloud import bigquery

from src.common.config import get_settings
```

### エラーハンドリング

- GCPクライアント呼び出しは `try/except` で `GoogleCloudError` をキャッチ
- ログ出力後に再 `raise` する
- 裸の `except:` は使用禁止（`except Exception:` を最低限使用）

## ファイル構成

- 1ファイル **500行以内**
- **単一責務**: 1ファイルには1つの明確な責務
- モジュール内ファイル構成例:
  ```
  src/data_collector/
  ├── __init__.py
  ├── client.py       # JRA API クライアント
  ├── parser.py       # レスポンスパーサー
  └── scheduler.py    # 収集スケジュール管理
  ```

## Git / ブランチ規則

### ブランチ戦略

- `main`: 本番環境。保護ブランチ。直接プッシュ禁止。
- `feature/*`: 機能開発（例: `feature/add-speed-features`）
- `fix/*`: バグ修正（例: `fix/bigquery-partition-error`）
- `docs/*`: ドキュメントのみの変更

### コミットメッセージ

**Conventional Commits** 形式を使用する:

```
<type>(<scope>): <description>

[optional body]
```

#### Type 一覧

| Type | 用途 |
|---|---|
| `feat` | 新機能の追加 |
| `fix` | バグ修正 |
| `docs` | ドキュメントのみの変更 |
| `refactor` | リファクタリング（機能変更なし） |
| `test` | テストの追加・修正 |
| `chore` | ビルド・設定変更など |
| `ci` | CI/CD設定の変更 |

#### 例

```
feat(data_collector): add JRA race data scraping
fix(feature_engineering): handle missing jockey data
docs: update setup guide with terraform steps
test(model_training): add unit tests for feature selection
chore: update LightGBM to v4.2.0
```

### PR 規則

- `feature/*` / `fix/*` ブランチから `main` への PR を作成
- **テスト必須**: PRのCIが全てパスすること
- **マージ方式**: Squash merge
- PRタイトルは Conventional Commits 形式で記述
- PR本文に以下を含める:
  - 変更概要
  - テスト内容
  - 関連Issue番号（該当する場合）

## コードレビュー基準

### 必須チェック項目

1. **型安全性**: すべての関数に型ヒントがあるか
2. **テスト**: 新機能・バグ修正にテストが追加されているか
3. **ドキュメント**: 公開APIにDocstringがあるか
4. **エラーハンドリング**: GCP呼び出しに適切な例外処理があるか
5. **コスト意識**: BigQueryクエリが不要なフルスキャンをしていないか
6. **セキュリティ**: シークレットがハードコードされていないか

### 推奨チェック項目

- 関数は小さく保たれているか（50行以内を目安）
- 変数名・関数名が明確か
- 不要なコメントがないか（コードで語る）
- Polars を Pandas より優先して使用しているか（パフォーマンス）

## テスト規約

### ディレクトリ構成

```
tests/
├── unit/               # ユニットテスト
│   ├── test_config.py
│   ├── test_gcp_client.py
│   └── ...
├── integration/        # 統合テスト（GCP接続あり）
│   └── ...
└── conftest.py         # 共通フィクスチャ
```

### テスト命名

- ファイル: `test_<module>.py`
- 関数: `test_<対象>_<条件>_<期待結果>`

```python
def test_upload_file_returns_gcs_uri() -> None:
    ...

def test_query_with_invalid_sql_raises_error() -> None:
    ...
```

### カバレッジ

- ユニットテストで **80%以上** を目標
- CIでカバレッジレポートを生成
