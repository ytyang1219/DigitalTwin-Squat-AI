# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   # 若需更抗離群值可改 RobustScaler
import joblib
import numpy as np

# ========= 使用者可調參數 =========
csv_path  = r"C:\Users\ytyan\squat_python\dataset_full.csv"
random_seed = 42
test_ratio = 0.2
val_ratio  = 0.2
fill_mode  = "fill"       # "fill" 或 "drop"
scaler_type = "standard"  # "standard" 或 "robust"

# 欄位定義
categ_cols = [
    'sex','level','RM_range','weekly_sets_range','quality_score_range',
    'frequency','fatigueL_desc','fatigueT_desc','labelT'
]
cont_cols  = ['target_weight','current_weight','target_ratio','target_weeks','week','recommend_weight']
label_col  = "label"

# ========= 1) 讀取與基本清理 =========
df = pd.read_csv(csv_path, skipinitialspace=True)

# 移除非特徵欄位
if "user_id" in df.columns:
    df = df.drop(columns=["user_id"])

# 缺失值處理
if fill_mode == "fill":
    for c in categ_cols:
        df[c] = df[c].fillna("unknown")
    for c in cont_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
else:
    df = df.dropna()

# label 轉為 0-based（假設原本是 1~5）
df[label_col] = pd.to_numeric(df[label_col], errors="coerce").astype(int)
df[label_col] = df[label_col] - df[label_col].min()

# ========= 2) 切分（先切再擬合前處理；分層抽樣） =========
train_val_df, test_df = train_test_split(
    df, test_size=test_ratio, random_state=random_seed, stratify=df[label_col]
)
val_ratio_adj = val_ratio / (1.0 - test_ratio)
train_df, val_df = train_test_split(
    train_val_df, test_size=val_ratio_adj, random_state=random_seed,
    stratify=train_val_df[label_col]
)

# ========= 3) 類別映射（僅用訓練集擬合；未知值有保護） =========
cat_maps = {}
for col in categ_cols:
    # 以訓練集的實際值建立映射（由 0 起算）
    cats = pd.Index(train_df[col].astype("category").cat.categories)
    mapping = {k: i for i, k in enumerate(cats, start=0)}
    cat_maps[col] = mapping

    def encode_with_train_map(s):
        # 未見過的值給 -1，稍後平移到合法索引
        return s.map(mapping).fillna(-1).astype(int)

    train_df[col] = encode_with_train_map(train_df[col])
    val_df[col]   = encode_with_train_map(val_df[col])
    test_df[col]  = encode_with_train_map(test_df[col])

# 將 -1（未知）統一平移到 0，並同步平移 mapping
for col in categ_cols:
    if min(train_df[col].min(), val_df[col].min(), test_df[col].min()) < 0:
        train_df[col] += 1; val_df[col] += 1; test_df[col] += 1
        cat_maps[col] = {k: (v + 1) for k, v in cat_maps[col].items()}

# ========= 4) 連續欄位標準化（僅用訓練集擬合） =========
if scaler_type == "robust":
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
else:
    scaler = StandardScaler()

train_df.loc[:, cont_cols] = scaler.fit_transform(train_df[cont_cols])
val_df.loc[:,   cont_cols] = scaler.transform(val_df[cont_cols])
test_df.loc[:,  cont_cols] = scaler.transform(test_df[cont_cols])

# ========= 5) 類別基數與摘要 =========
categories = tuple(int(max(train_df[c].max(), val_df[c].max(), test_df[c].max()) + 1) for c in categ_cols)
num_continuous = len(cont_cols)
num_classes    = int(train_df[label_col].nunique())

# ========= 6) 輸出三個 split 與前處理 meta =========
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
print("CSV 已輸出：train.csv, val.csv, test.csv")

preprocess_meta = {
    "random_seed": random_seed,
    "categ_cols": categ_cols,
    "cont_cols": cont_cols,
    "label_col": label_col,
    "categories": categories,
    "num_continuous": num_continuous,
    "num_classes": num_classes,
    "scaler_type": scaler_type,
    "scaler_mean": getattr(scaler, "mean_", None).tolist() if hasattr(scaler, "mean_") else None,
    "scaler_scale": getattr(scaler, "scale_", None).tolist() if hasattr(scaler, "scale_") else None,
    "cat_maps": cat_maps
}
joblib.dump(preprocess_meta, "preprocess_meta.pkl")
print("前處理參數已儲存：preprocess_meta.pkl")




# === 4. One-hot 編碼（供 MLP 使用）===
# def one_hot_df(df, categ_cols, cont_cols):
#     label = df["label"]
#     df = df.drop(columns=["label"])
#     df = pd.get_dummies(df, columns=categ_cols)
#     df["label"] = label
#     return df

# train_df_1hot = one_hot_df(train_df_raw, categ_cols, cont_cols).astype(int)
# val_df_1hot = one_hot_df(val_df_raw, categ_cols, cont_cols).astype(int)
# test_df_1hot = one_hot_df(test_df_raw, categ_cols, cont_cols).astype(int)

# def align_columns(df, all_cols):
#     for col in all_cols:
#         if col not in df.columns:
#             df[col] = 0
#     return df[all_cols]

# # 對齊欄位（確保欄位一致）
# all_cols = sorted(set(train_df_1hot.columns) | set(val_df_1hot.columns) | set(test_df_1hot.columns))
# train_df_1hot = align_columns(train_df_1hot, all_cols)
# val_df_1hot = align_columns(val_df_1hot, all_cols)
# test_df_1hot = align_columns(test_df_1hot, all_cols)

# # 補齊 one-hot 缺失值
# train_df_1hot = train_df_1hot.fillna(0)
# val_df_1hot = val_df_1hot.fillna(0)
# test_df_1hot = test_df_1hot.fillna(0)

# # === 5. 儲存 One-hot 編碼版本 ===
# train_df_1hot.to_csv("train_mlp.csv", index=False)
# val_df_1hot.to_csv("val_mlp.csv", index=False)
# test_df_1hot.to_csv("test_mlp.csv", index=False)
# print("MLP One-hot 資料儲存完畢：train_mlp.csv, val_mlp.csv, test_mlp.csv")