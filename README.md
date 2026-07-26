# anochan

`anochan`は、`malchan`に含まれていた異常検知処理を独立して扱うためのPythonパッケージです。初期段階では、`malchan`の構成にできるだけ合わせて、**前処理と異常検知モデルを1つのscikit-learn Pipelineへまとめること**を優先しています。

時系列window、装置・ライン単位の系列分割、時間集約、変化点検知は今回の範囲に含めません。

## Pipeline構成

学習後の`model.model`は次の2ステップを持ちます。

```text
Pipeline
├── preprocess
│   ├── 数値列: 欠損補完・スケーリング
│   ├── カテゴリ列: 欠損補完・One-Hot Encoding
│   ├── 多項式・交互作用特徴量（任意）
│   └── 次元削減（任意）
└── predictor
    └── 異常検知モデル
```

これは`malchan.models.pipelines.make_pipeline()`の`preprocess`→`predictor`構成に合わせています。教師なし異常検知では不要な`target_col`、`task`、チューニング、アンサンブル関連の引数は除外しています。

## 基本的な使用方法

```python
from anochan import AnomalyDetectionPipeline

model = AnomalyDetectionPipeline()
model.fit(
    df,
    num_cols=["temperature", "current", "pressure"],
    cat_cols=["machine"],
    model_names=["IsolationForest"],
    model_params={"random_state": 42},
    num_impute_type="median",
    num_scale_type="StandardScaler",
    cat_impute=True,
)

result = model.predict(new_df)
```

`model_names`は`malchan`と同じリスト形式ですが、異常検知では1モデルだけを指定します。

`model.model`は通常のscikit-learn Pipelineなので、各ステップへ直接アクセスできます。

```python
preprocess = model.model.named_steps["preprocess"]
predictor = model.model.named_steps["predictor"]

preprocessed_df = model.transform(new_df)
```

## 入力列

目的変数は使用しません。入力列は`malchan`と同様に数値列とカテゴリ列へ分けて指定します。

```python
num_cols=["temperature", "current"]
cat_cols=["machine", "product"]
```

`feature_cols`、`exclude_cols`、`target_col`、`target_cols`は使用しません。

## 数値前処理

`num_impute_type`:

- `None`
- `Multiple`
- `mean`
- `median`
- `most_frequent`
- `knn`

`num_scale_type`:

- `None`
- `StandardScaler`
- `MinMaxScaler`
- `centering`
- `MaxAbsScaler`

## カテゴリ前処理

`cat_cols`はOne-Hot Encodingされます。`cat_impute=True`の場合、欠損値を最頻値で補完してからエンコードします。未知カテゴリは無視されるため、学習時に存在しないカテゴリを含むデータも変換できます。

## 特徴量変換

```python
model.fit(
    df,
    num_cols=["x1", "x2", "x3"],
    model_names=["OneClassSVM"],
    num_scale_type="StandardScaler",
    poly=True,
    poly_degree=2,
    poly_interaction_only=True,
    decomposition=True,
    decomposition_method="PCA",
    dec_n_components=2,
)
```

次元削減は`PCA`、`KernelPCA`、`NMF`、`ICA`に対応します。

## 異常検知モデル

`malchan`の異常検知モデル定義から次の3モデルを移しています。

| `model_names`の要素 | 既定値 |
|---|---|
| `OneClassSVM` | `nu=0.2`, `kernel="rbf"`, `gamma="auto"` |
| `IsolationForest` | `n_estimators=100`, `contamination="auto"` |
| `EllipticEnvelope` | `contamination=0.01` |

既定値は`model_params`で上書きできます。

```python
model.fit(
    df,
    num_cols=["x1", "x2"],
    model_names=["IsolationForest"],
    model_params={
        "n_estimators": 300,
        "contamination": 0.03,
        "random_state": 42,
    },
)
```

## 出力

```python
result = model.predict(df)
```

| 列 | 内容 |
|---|---|
| `prediction` | scikit-learn準拠。正常=`1`、異常=`-1` |
| `is_anomaly` | `prediction == -1`の真偽値 |
| `decision_function` | モデル標準の判別値。大きいほど正常側 |
| `anomaly_score` | `-decision_function`。大きいほど異常側 |

## 保存と読み込み

前処理と異常検知モデルをまとめて保存します。

```python
model.save("models/anomaly_pipeline.joblib")
loaded = AnomalyDetectionPipeline.load("models/anomaly_pipeline.joblib")
result = loaded.predict(new_df)
```

## FastAPIでの提供

FastAPI関連は任意依存としてインストールします。

```bash
pip install -e ".[api]"
```

学習済みPipelineを保存してからAPIを起動します。

```python
model.save("models/anomaly_pipeline.joblib")
```

```bash
anochan-api \
  --model-path models/anomaly_pipeline.joblib \
  --host 0.0.0.0 \
  --port 8000
```

Windowsのコマンドプロンプトでは1行で実行できます。

```bat
anochan-api --model-path models\anomaly_pipeline.joblib --host 0.0.0.0 --port 8000
```

環境変数を使用する場合は、`ANOCHAN_MODEL_PATH`を設定してUvicornを直接起動できます。

```bat
set ANOCHAN_MODEL_PATH=models\anomaly_pipeline.joblib
uvicorn anochan.api:app --host 0.0.0.0 --port 8000
```

提供するエンドポイントは次のとおりです。

| メソッド | パス | 内容 |
|---|---|---|
| `GET` | `/health` | モデル読込状態。未読込時はHTTP 503 |
| `GET` | `/v1/model` | 必須列、数値列、カテゴリ列、モデル設定 |
| `POST` | `/v1/predict` | 前処理と異常判定をまとめて実行 |
| `POST` | `/v1/transform` | 前処理後の特徴量を確認 |

### 推論リクエスト

```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "temperature": 800.0,
        "current": 12.0,
        "pressure": 1.02,
        "machine": "A"
      },
      {
        "temperature": 860.0,
        "current": 18.0,
        "pressure": 1.30,
        "machine": "B"
      }
    ]
  }'
```

レスポンス例:

```json
{
  "model_name": "IsolationForest",
  "count": 2,
  "predictions": [
    {
      "row_index": 0,
      "prediction": 1,
      "is_anomaly": false,
      "decision_function": 0.12,
      "anomaly_score": -0.12
    },
    {
      "row_index": 1,
      "prediction": -1,
      "is_anomaly": true,
      "decision_function": -0.08,
      "anomaly_score": 0.08
    }
  ]
}
```

学習時の必須列が不足している場合はHTTP 422、モデルが読み込まれていない場合はHTTP 503、1リクエストの上限件数を超えた場合はHTTP 413を返します。既定の上限は10,000行です。

Pythonアプリケーションへ組み込む場合は、保存済みモデルまたは読込済みPipelineを渡します。

```python
from anochan.api import create_app

app = create_app(
    model_path="models/anomaly_pipeline.joblib",
    max_batch_size=1000,
)
```

```python
from anochan.api import create_app

app = create_app(model=model)
```

モデルはFastAPIのlifespanで起動時に1回だけ読み込まれます。`--workers`を増やした場合は、ワーカープロセスごとにモデルが1つずつ読み込まれるため、モデルサイズに応じてメモリ使用量を確認してください。

`joblib`ファイルはPythonのpickle形式を利用するため、信頼できる環境で作成したモデルファイルだけを読み込んでください。

FastAPI起動後は、`/docs`でSwagger UI、`/redoc`でReDocを確認できます。

## 今回含めないもの

`malchan`の材料・化学向けSMILES／組成特徴量生成は重い任意依存を必要とするため、今回の表形式異常検知の分離には含めていません。必要になった段階で、独立したoptional dependencyとして追加します。
