# anochan

`anochan`は、`malchan`に含まれていた異常検知処理を独立して扱うためのPythonパッケージです。`malchan`の構成にできるだけ合わせて、**前処理と異常検知モデルを1つのscikit-learn Pipelineへまとめること**を優先しています。

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

`malchan`由来の3モデルに加えて、同じ`fit / predict / decision_function` APIで扱える5モデルを追加しています。

| `model_names`の要素 | 検知の考え方 | 主な用途 | 主な既定値 |
|---|---|---|---|
| `OneClassSVM` | カーネル境界 | 非線形な正常領域 | `nu=0.2`, `kernel="rbf"` |
| `IsolationForest` | ランダム分割 | 汎用、大規模データ | `n_estimators=100` |
| `EllipticEnvelope` | 頑健共分散 | 楕円状・正規分布に近い正常データ | `contamination=0.01` |
| `LocalOutlierFactor` | 局所密度 | 正常クラスタが複数あるデータ | `n_neighbors=20`, `novelty=True` |
| `SGDOneClassSVM` | 線形境界の確率的最適化 | 高次元・大量データ | `nu=0.05` |
| `KNN` | 近傍までの平均距離 | 局所的に孤立した点 | `n_neighbors=5`, `contamination=0.05` |
| `PCAReconstruction` | PCA再構成誤差 | 相関した多変量プロセス | `n_components=0.95` |
| `GaussianMixture` | 混合正規分布の尤度 | 複数の運転モード・密度異常 | `n_components=1`, `contamination=0.05` |

利用可能なモデル名は次のように確認できます。

```python
print(AnomalyDetectionPipeline.available_models())
```

既定値は`model_params`で上書きできます。

```python
model.fit(
    df,
    num_cols=["x1", "x2"],
    model_names=["LocalOutlierFactor"],
    model_params={
        "n_neighbors": 30,
        "contamination": 0.03,
    },
    num_scale_type="StandardScaler",
)
```

`LocalOutlierFactor`は学習後の新規データを判定できるよう、`novelty=True`を必須としています。

### モデル選択の目安

- 最初の基準モデル: `IsolationForest`
- 正常領域の境界が非線形: `OneClassSVM`
- データ量・特徴量数が多い: `SGDOneClassSVM`
- 局所的な密度低下を見たい: `LocalOutlierFactor`または`KNN`
- センサ間相関からの崩れを見たい: `PCAReconstruction`
- 正常状態が複数の分布で表せる: `GaussianMixture`
- 正常データが単峰の楕円分布に近い: `EllipticEnvelope`

距離・境界・密度を利用するモデルでは、通常`num_scale_type="StandardScaler"`を推奨します。

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

## 今回含めないもの

`malchan`の材料・化学向けSMILES／組成特徴量生成は重い任意依存を必要とするため、今回の表形式異常検知の分離には含めていません。必要になった段階で、独立したoptional dependencyとして追加します。
