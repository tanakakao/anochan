# 時系列異常検知Pipeline

`TimeSeriesAnomalyDetectionPipeline`は、時系列固有のwindow生成を担当し、前処理と異常検知モデルの学習は既存の`AnomalyDetectionPipeline`へ委譲します。

```text
TimeSeriesAnomalyDetectionPipeline
├── 時刻変換・並べ替え
├── group_colsごとの系列分離
├── window特徴量生成
├── 右端時刻との対応管理
└── AnomalyDetectionPipeline
    ├── preprocess
    └── predictor
```

`TimeSeriesAnomalyDetectionPipeline`は`AnomalyDetectionPipeline`を継承していません。入力行と出力行の対応が異なるため、内部に保持するコンポジション構成です。

## 基本的な使用方法

```python
from anochan import TimeSeriesAnomalyDetectionPipeline

model = TimeSeriesAnomalyDetectionPipeline().fit(
    df,
    time_col="timestamp",
    group_cols=["machine"],
    num_cols=["temperature", "current"],
    cat_cols=["mode"],
    window_size=5,
    stride=1,
    window_features=["raw", "diff", "mean", "std"],
    model_names=["IsolationForest"],
    model_params={
        "n_estimators": 300,
        "contamination": 0.03,
        "random_state": 42,
    },
    num_impute_type="median",
    num_scale_type="StandardScaler",
    cat_impute=True,
)

result = model.predict(new_df)
```

## windowの扱い

- `time_col`をdatetimeへ変換し、各グループ内で昇順に並べます。
- `group_cols`が指定されている場合、windowはグループ境界を跨ぎません。
- `window_size`未満のグループはwindowを生成せず、他の有効なグループだけを利用します。
- すべてのグループが短い場合は`ValueError`を返します。
- 判定結果はwindowの最後の観測へ割り当てます。初期実装のalignmentは`right`固定です。
- `stride=1`では1行ずつwindowを進めます。`stride=2`では2行ずつ進めます。

## window特徴量

`window_features`には次を組み合わせて指定できます。

| 値 | 生成内容 |
|---|---|
| `raw` | window内の値を右端基準のlag特徴量へ展開 |
| `diff` | 隣接時点の差分を右端基準で展開 |
| `mean` | window平均 |
| `std` | window標準偏差（`ddof=0`） |
| `min` | window最小値 |
| `max` | window最大値 |

`window_size=3`、`num_cols=["temperature"]`、`window_features=["raw", "diff", "mean"]`の場合、次の数値特徴量を生成します。

```text
temperature__lag_0
temperature__lag_1
temperature__lag_2
temperature__diff_lag_0
temperature__diff_lag_1
temperature__mean
```

`lag_0`はwindow右端の現在値、`lag_1`は1つ前、`lag_2`は2つ前です。`diff_lag_0`は現在値と1つ前の差です。

`cat_cols`はwindow右端の値を使用し、既存のカテゴリ前処理へ渡します。`group_cols`は系列分離と出力メタデータに使用し、`cat_cols`にも指定しない限りモデル入力には含めません。

## 出力

`predict()`はwindowごとに次を返します。

| 列 | 内容 |
|---|---|
| `group_cols` | windowが属する装置・ラインなど |
| `time_col` | window右端の時刻 |
| `window_start_time` | window開始時刻 |
| `window_end_time` | window終了時刻 |
| `window_start_index` | 元DataFrameの開始行index |
| `window_end_index` | 元DataFrameの終了行index |
| `prediction` | 正常=`1`、異常=`-1` |
| `is_anomaly` | `prediction == -1` |
| `decision_function` | 大きいほど正常側 |
| `anomaly_score` | 大きいほど異常側 |

生成されたwindow特徴量は次で確認できます。

```python
window_df = model.make_windows(df)
```

既存前処理を適用した後の特徴量は次で確認できます。

```python
preprocessed_df = model.transform(df)
```

## 保存と読み込み

window設定、前処理器、異常検知モデルをまとめて保存します。

```python
model.save("models/time_series_anomaly_pipeline.joblib")

loaded = TimeSeriesAnomalyDetectionPipeline.load(
    "models/time_series_anomaly_pipeline.joblib"
)
result = loaded.predict(new_df)
```

## FastAPI

保存済み時系列モデルは、表形式モデルと同じCLIで起動できます。

```bash
anochan-api \
  --model-path models/time_series_anomaly_pipeline.joblib \
  --host 0.0.0.0 \
  --port 8000
```

`POST /v1/predict`へ送る`records`には、少なくとも各グループについて`window_size`件の履歴が必要です。レスポンスでは時刻、グループ、window範囲を各予測の`metadata`へ格納します。

```json
{
  "pipeline_type": "time_series",
  "model_name": "IsolationForest",
  "count": 1,
  "predictions": [
    {
      "row_index": 0,
      "prediction": 1,
      "is_anomaly": false,
      "decision_function": 0.12,
      "anomaly_score": -0.12,
      "metadata": {
        "machine": "A",
        "timestamp": "2026-01-01T04:00:00",
        "window_start_time": "2026-01-01T00:00:00",
        "window_end_time": "2026-01-01T04:00:00",
        "window_start_index": 0,
        "window_end_index": 4
      }
    }
  ]
}
```

現在のAPIはリクエスト内のデータだけでwindowを作成するステートレス方式です。逐次的に1行ずつ送信して内部履歴を保持するリアルタイム監視は、履歴ストアや状態管理を含む別機能として扱います。

## 初期実装に含まれない機能

- 時間単位のresample・集約
- 時間間隔の不均一性や欠測区間の検査
- 左端・中央へのwindow alignment
- リクエストを跨ぐ履歴バッファ
- VAR、ARIMA、状態空間モデルなどのネイティブ時系列モデル
- 変化点検知

これらはwindow型の表形式異常検知と責務が異なるため、段階的に追加します。
