# システムアーキテクチャ

## GCPサービス構成図

```
                        +-------------------+
                        |   GitHub Actions   |
                        | (CI/CD Pipeline)   |
                        +--------+----------+
                                 |
                    Workload Identity Federation
                                 |
                                 v
+----------------------------------------------------------------+
|                        Google Cloud Platform                     |
|                                                                  |
|  +------------------+    +------------------+  +------------------+|
|  | Secret Manager   |    | Cloud Run Jobs   |  | Cloud Run Service||
|  | - jra-api-key    |    | - ml-pipeline    |  | - mlflow-ui-{env}||
|  +--------+---------+    | (バッチ実行)      |  | (HTTP Tracking)  ||
|           |              +----+----+----+---+  +--------+---------+|
|           |                   |    |    |                         |
|           v                   |    |    |                         |
|  +------------------+         |    |    |                         |
|  | Cloud Functions  |         |    |    |                         |
|  | (Data Collector) +---------+    |    |                         |
|  +--------+---------+              |    |                         |
|           |                        |    |                         |
|           v                        v    v                         |
|  +------------------+    +------------------+                    |
|  | Cloud Storage    |    | BigQuery         |                    |
|  | (GCS)            |    |                  |                    |
|  | +- raw-data      |    | +- races_raw     |                    |
|  | +- processed     |    | +- features      |                    |
|  | +- models        |    | +- predictions   |                    |
|  +------------------+    +------------------+                    |
|                                                                  |
+----------------------------------------------------------------+
```

## データフロー図

```
[JRA API / Web]
       |
       v
+------+--------+     +-----------+     +----------+     +----------+
| data_collector | --> | GCS       | --> | BigQuery | --> | BigQuery |
| (収集)         |     | raw-data  |     | races_raw|     | features |
+----------------+     +-----------+     +----+-----+     +----+-----+
                                              |                 |
                                              v                 v
                                    +---------+-------+  +------+--------+
                                    | feature_        |  | model_        |
                                    | engineering     |  | training      |
                                    | (特徴量生成)     |  | (学習)         |
                                    +-----------------+  +------+--------+
                                                                |
                                                                v
                                                         +------+--------+
                                                         | GCS           |
                                                         | models/       |
                                                         | (学習済みモデル) |
                                                         +------+--------+
                                                                |
                                                                v
                                                         +------+--------+
                                                         | predictor     |
                                                         | (予測)         |
                                                         +------+--------+
                                                                |
                                                                v
                                                         +------+--------+
                                                         | BigQuery      |
                                                         | predictions   |
                                                         +------+--------+
                                                                |
                                                                v
                                                         +------+--------+
                                                         | evaluator     |
                                                         | (評価)         |
                                                         +---------------+
```

### パイプライン詳細

1. **データ収集** (`data_collector`)
   - JRA APIからレースデータを取得
   - 生データをGCS (`raw-data` バケット) に保存
   - パース後のデータをBigQuery (`races_raw` テーブル) に格納
   - 日次パーティション: `race_date`

2. **特徴量生成** (`feature_engineering`)
   - BigQuery `races_raw` から生データを読み込み
   - 馬・騎手・コース等の特徴量を計算
   - BigQuery `features` テーブルに格納
   - クラスタリングキー: `feature_version`

3. **モデル学習** (`model_training`)
   - BigQuery `features` テーブルからデータを取得
   - LightGBM で分類/回帰モデルを学習
   - MLflow HTTP Tracking Server 経由で実験・パラメータ・メトリクス・アーティファクトを記録
   - `RequestHeaderProvider` プラグインが GCP OIDC ID トークンを自動付与（Cloud Run IAM 認証）
   - 学習済みモデルをGCS (`models` バケット) に保存

4. **予測** (`predictor`)
   - GCS から学習済みモデルをロード
   - BigQuery `features` から当日のデータを取得
   - 予測結果をBigQuery `predictions` テーブルに格納

5. **評価** (`evaluator`)
   - BigQuery `predictions` と実際のレース結果を突合
   - 精度指標・回収率等を計算
   - 結果をレポートとして出力

## モジュール連携図

```
src/
├── common/                     [共通基盤]
│   ├── config.py               設定管理 (Pydantic Settings + YAML)
│   ├── gcp_client.py           GCS/BigQueryクライアント
│   ├── logging.py              structlogロギング
│   └── mlflow_auth.py          MLflow Cloud Run IAM認証プラグイン
│
├── data_collector/             [データ収集]
│   ├── client.py               JRA APIクライアント
│   ├── parser.py               レスポンスパーサー
│   └── scheduler.py            スケジュール管理
│         |
│         | uses: common.gcp_client (GCS, BigQuery)
│         | uses: common.config (API設定)
│         v
├── feature_engineering/        [特徴量生成]
│   ├── pipeline.py             特徴量パイプライン
│   ├── features/               個別特徴量定義
│   └── transformer.py          データ変換
│         |
│         | uses: common.gcp_client (BigQuery)
│         | reads: races_raw -> writes: features
│         v
├── model_training/             [モデル学習]
│   ├── trainer.py              学習ループ
│   ├── model.py                モデル定義 (LightGBM)
│   └── experiment.py           MLflow実験管理
│         |
│         | uses: common.gcp_client (GCS, BigQuery)
│         | reads: features -> writes: GCS models/
│         v
├── predictor/                  [予測]
│   ├── __main__.py             エントリポイント
│   ├── predictor.py            予測ロジック
│   └── loader.py               モデルローダー
│         |
│         | uses: common.gcp_client (GCS, BigQuery)
│         | reads: features, models/ -> writes: predictions
│         v
└── evaluator/                  [評価]
    ├── backtester.py           バックテスト
    ├── metrics.py              精度メトリクス
    └── report.py               レポート生成
          |
          | uses: common.gcp_client (BigQuery)
          | reads: predictions, races_raw
```

## コスト試算表

### 月間コスト想定（通常運用時）

| サービス | 使用量想定 | 単価 | 月額コスト |
|---|---|---|---|
| **Cloud Storage** | | | |
| - raw-data | ~100MB | $0.020/GB/月 | ~$0.002 |
| - processed | ~50MB | $0.020/GB/月 | ~$0.001 |
| - models | ~10MB (versioned) | $0.020/GB/月 | ~$0.001 |
| - Nearline (90日後) | ~500MB | $0.010/GB/月 | ~$0.005 |
| **BigQuery** | | | |
| - ストレージ | ~1GB (最初の10GB無料) | $0/月 | $0 |
| - クエリ | ~5GB/月 (最初の1TB無料) | $0/月 | $0 |
| **Cloud Run Jobs** | | | |
| - 実行回数 | ~30回/月 (Free tier内) | $0/月 | $0 |
| - CPU | ~10 CPU時間 (180,000 vCPU秒無料) | $0/月 | $0 |
| - メモリ | 512Mi x 10時間 (360,000 GiB秒無料) | $0/月 | $0 |
| **Cloud Run Service** | | | |
| - MLflow UI | min-instances=0, スケールゼロ | $0/月 | ~$0 |
| **Cloud Functions** | | | |
| - 呼び出し | ~30回/月 (200万回無料) | $0/月 | $0 |
| **Secret Manager** | | | |
| - アクセス | ~100回/月 (10,000回無料) | $0/月 | $0 |
| | | **合計** | **~$0.01/月** |

### コスト最適化戦略

- **GCS ライフサイクルルール**: 90日経過後にNearline Storage Classへ自動移行
- **BigQuery パーティション**: `race_date` による日次パーティション。クエリが必要な日付範囲のみスキャン
- **BigQuery クラスタリング**: `feature_version` でクラスタリング。特定バージョンのみ読み取り
- **Cloud Run Jobs**: バッチ実行時のみ課金。実行していない間はコストゼロ
- **Models バケット**: バージョニング有効で履歴管理（不要な古いバージョンは定期削除）

## 拡張パス

### Phase 1: 手動運用（現在）

```
手動実行 -> データ収集 -> 特徴量生成 -> 学習 -> 予測 -> 評価
```

- CLIまたはCloud Run Jobsでパイプラインを手動実行
- 結果をBigQueryで確認
- モデルの改善サイクルを手動で回す

### Phase 2: 自動化

```
Cloud Scheduler -> Cloud Functions (トリガー) -> Cloud Run (パイプライン)
```

- Cloud Scheduler で日次/週次の自動実行
- Cloud Functions がCloud Run Jobsをトリガー
- Pub/Sub でパイプラインステージ間を連携
- エラー通知（Cloud Monitoring + アラート）

### Phase 3: リアルタイム対応

```
リアルタイムデータ -> Pub/Sub -> Cloud Run -> 予測API -> 結果表示
```

- レース開催日にリアルタイムでオッズ変動を取得
- Pub/Sub でストリーミング処理
- 予測結果をREST APIとして公開
- ダッシュボード（Looker Studio等）で可視化

### 将来的な技術拡張

| 領域 | 現在 | 将来 |
|---|---|---|
| モデル | LightGBM | + Neural Network (PyTorch) |
| 特徴量 | バッチ計算 | + リアルタイム特徴量 |
| データソース | JRA API | + 海外レース、ニュース |
| 可視化 | BigQuery SQL | Looker Studio ダッシュボード |
| モニタリング | ログ | Cloud Monitoring + アラート |
