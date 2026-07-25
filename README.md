# anochan

`anochan` は、製造プロセス・センサ・表形式データを対象とする独立した異常検知フレームワークです。`malchan` の教師あり機械学習ワークフローから異常検知を分離し、教師なし異常検知に必要な入力・学習・異常度・しきい値管理だけを扱います。

## 設計方針

- **目的変数は不要**: `target_col` / `target_cols` / `targetcols` はAPIに存在しません。
- **DataFrame-first**: 特徴量、時刻列、グループ列を明示して学習・判定します。
- **時系列window対応**: 過去から現在までの値を1行へ展開し、周期的・遷移的な異常を検知します。
- **グループ境界を保護**: 装置・ライン・品種など、異なる系列をまたいだwindowを作りません。
- **異常度としきい値を分離**: 学習後もモデルを再学習せずにしきい値を変更できます。
- **`malchan`は変更しない**: 本リポジトリは単独でインストール・保存・運用できます。

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
    time_col="timestamp",
    group_cols=["machine"],
    window_size=5,
)

output = pd.concat([df, result], axis=1)
print(output[["timestamp", "anomaly_score", "is_anomaly"]])
```

`window_size=5`では、同一グループ内の5時点を `[t-4, t-3, t-2, t-1, t]` の順で平坦化します。各グループの先頭4行は完全なwindowを作れないため、`anomaly_score`と`is_anomaly`は欠損になります。

## 特徴量の自動選択

`feature_cols=[]`または未指定の場合、数値列から時刻列・グループ列・`exclude_cols`を除いて特徴量を推定します。既知の異常ラベル、ID、バッチ番号などの数値メタデータは`exclude_cols`へ指定してください。

```python
model.fit(
    df,
    exclude_cols=["known_label", "batch_no"],
    time_col="timestamp",
    group_cols=["machine", "product"],
)
```

教師なし異常検知であるため、除外列を目的変数として学習することはありません。

## 利用可能な手法

| detector | 異常度の考え方 | 主な用途 |
|---|---|---|
| `robust_zscore` | 中央値・MADからの頑健な乖離 | 単純監視、外れ値、説明しやすさ重視 |
| `pca` | Q残差とHotelling T² | 多変量プロセス、相関したセンサ |
| `knn` | 近傍点までの平均距離 | 局所的な疎領域の検出 |
| `lof` | 局所密度の低下 | 複数の正常クラスタを含むデータ |
| `isolation_forest` | 分離されやすさ | 汎用的な非線形異常検知 |
| `one_class_svm` | 正常領域境界からの逸脱 | 正常領域が比較的明確な場合 |
| `elliptic_envelope` | 頑健マハラノビス距離 | 楕円状・正規分布に近い正常データ |
| `kmeans` | 最近傍クラスタ中心からの距離 | 運転モードが複数ある場合 |
| `dbscan` | 最近傍コアサンプルからの距離 | 任意形状の正常クラスタ |
| `graphical_lasso` | 疎な精度行列に基づく関係逸脱 | センサ間・時点間の相関監視 |

```python
print(AnomalyDetectionPipeline.available_detectors())
```

## Graphical Lassoとwindow

`graphical_lasso`へ`window_size > 1`を指定すると、精度行列は現在値だけでなく過去時点を含む特徴量間関係を学習します。異常度は、学習した疎な関係構造に対するprecision-weightedな逸脱度です。

```python
model = AnomalyDetectionPipeline(
    detector="graphical_lasso",
    detector_params={"alpha": 0.05},
    contamination=0.02,
)
model.fit(
    df,
    feature_cols=["sensor_a", "sensor_b", "sensor_c"],
    time_col="timestamp",
    group_cols=["equipment"],
    window_size=10,
)
```

## 学習後のしきい値変更

異常度を再計算・再学習せず、判定しきい値だけを変更できます。

```python
scores = model.score_samples(df)

model.set_threshold(contamination=0.01)
result_1pct = model.predict(df)

model.set_threshold(threshold=8.5)
result_fixed = model.predict(df)
```

一時的なしきい値で判定する場合はモデル状態を変更しません。

```python
result = model.predict(df, threshold=7.0)
```

## 保存と読み込み

```python
model.save("models/furnace_anomaly.joblib")
loaded = AnomalyDetectionPipeline.load("models/furnace_anomaly.joblib")
result = loaded.predict(new_df)
```

## カスタム検知器

`AnomalyDetector`を継承し、`fit()`と`score_samples()`を実装できます。`score_samples()`は、**大きいほど異常**となる1次元スコアを返してください。しきい値管理、DataFrame整列、欠損補完、window化、保存は`AnomalyDetectionPipeline`が担当します。
