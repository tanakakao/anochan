# anochan

`anochan` は、`malchan` から異常検知部分を分離して扱うための独立した Python パッケージです。今回の初期実装では、表形式データの各行を1つの観測として扱う純粋な異常検知に範囲を限定しています。

時系列window、装置・ライン別の系列分割、時間集約などは含めません。これらは、基本的な異常検知機能を分離した後の拡張として追加します。

## 今回の分離方針

- `malchan` 側は変更しない
- 教師なし異常検知のため、`target_col` / `target_cols` / `targetcols` は使用しない
- 入力は `feature_cols` で指定する
- `feature_cols` 未指定時は数値列を自動選択する
- ラベル、ID、バッチ番号などの数値メタデータは `exclude_cols` で除外する
- 欠損値補完、標準化、異常度計算、しきい値判定、モデル保存を提供する
- 時系列固有の引数や処理は追加しない

## インストール

```bash
git clone https://github.com/tanakakao/anochan.git
cd anochan
pip install -e .
```

開発用:

```bash
pip install -e ".[dev]"
```

## 基本例

```python
import pandas as pd

from anochan import AnomalyDetectionPipeline

model = AnomalyDetectionPipeline(
    detector="isolation_forest",
    contamination=0.03,
)

result = model.fit_predict(
    df,
    feature_cols=["temperature", "current", "pressure"],
)

output = pd.concat([df, result], axis=1)
print(output[["anomaly_score", "threshold", "is_anomaly"]])
```

各入力行に対して、次の3列を返します。

| 列 | 内容 |
|---|---|
| `anomaly_score` | 異常度。大きいほど異常 |
| `threshold` | 判定に使用したしきい値 |
| `is_anomaly` | 異常判定結果 |

## 特徴量の自動選択

`feature_cols`を指定しない場合は数値列を自動選択します。既知の異常ラベル、ID、バッチ番号など、特徴量に使用しない数値列は`exclude_cols`へ指定します。

```python
model.fit(
    df,
    exclude_cols=["known_label", "batch_no"],
)
```

非数値列は自動選択の対象になりません。

## 利用可能な手法

| detector | 異常度の考え方 |
|---|---|
| `robust_zscore` | 中央値・MADからの頑健な乖離 |
| `pca` | Q残差とHotelling T² |
| `knn` | 近傍点までの平均距離 |
| `lof` | 局所密度の低下 |
| `isolation_forest` | 分離されやすさ |
| `one_class_svm` | 正常領域境界からの逸脱 |
| `elliptic_envelope` | 頑健マハラノビス距離 |
| `kmeans` | 最近傍クラスタ中心からの距離 |
| `dbscan` | 最近傍コアサンプルからの距離 |
| `graphical_lasso` | 疎な多変量関係からの逸脱 |

```python
print(AnomalyDetectionPipeline.available_detectors())
```

## 学習後のしきい値変更

検知器を再学習せず、しきい値だけを変更できます。

```python
model.set_threshold(contamination=0.01)
result_1pct = model.predict(df)

model.set_threshold(threshold=8.5)
result_fixed = model.predict(df)
```

一時的なしきい値で判定する場合は、モデル状態を変更しません。

```python
result = model.predict(df, threshold=7.0)
```

## 保存と読み込み

```python
model.save("models/anomaly_model.joblib")
loaded = AnomalyDetectionPipeline.load("models/anomaly_model.joblib")
result = loaded.predict(new_df)
```

## 時系列対応について

現時点では、行順序に意味を持たせず、各行を独立した観測として処理します。時系列window、グループ別系列処理、変化点検知などは、この基本実装とは分けて後続対応します。
