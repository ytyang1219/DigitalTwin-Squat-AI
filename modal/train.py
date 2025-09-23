# -*- coding: utf-8 -*-
import time
import joblib
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 你的 FT-Transformer 實作
from demo.modal.ft_transformer import FTTransformer
# 若要測 TabTransformer，可自行引入：
# from tab_transformer_pytorch import TabTransformer

# ========= 1) 載入 split 輸出的資料與前處理 meta =========
train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")
test_df  = pd.read_csv("test.csv")
meta     = joblib.load("preprocess_meta.pkl")

categ_cols     = meta["categ_cols"]
cont_cols      = meta["cont_cols"]
label_col      = meta["label_col"]
categories     = tuple(meta["categories"])
num_continuous = int(meta["num_continuous"])
num_classes    = int(meta["num_classes"])

# ========= 2) Dataset / DataLoader =========
class TabDataset(Dataset):
    def __init__(self, df, categ_cols, cont_cols, label_col):
        self.y  = torch.tensor(df[label_col].values, dtype=torch.long)
        self.xc = torch.tensor(df[categ_cols].values, dtype=torch.long)
        self.xn = torch.tensor(df[cont_cols].values, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.xc[idx], self.xn[idx], self.y[idx]

train_loader = DataLoader(TabDataset(train_df, categ_cols, cont_cols, label_col), batch_size=32, shuffle=True)
val_loader   = DataLoader(TabDataset(val_df,   categ_cols, cont_cols, label_col), batch_size=32)
test_loader  = DataLoader(TabDataset(test_df,  categ_cols, cont_cols, label_col), batch_size=32)

# ========= 3) 建立模型 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FTTransformer(
    categories=categories,
    num_continuous=num_continuous,
    dim=32,
    depth=3,
    heads=4,
    dim_out=num_classes
).to(device)

# 若要比較 TabTransformer，可切換如下（記得維持相同 categories/num_continuous）：
# model = TabTransformer(
#     categories=categories,
#     num_continuous=num_continuous,
#     dim=32, depth=3, heads=4, dim_out=num_classes
# ).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ========= 4) 訓練 =========
num_epochs = 20
train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

start_time = time.time()
best_val_loss = float("inf")
best_state = None

for epoch in range(1, num_epochs + 1):
    # ---- train ----
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for xc, xn, y in train_loader:
        xc, xn, y = xc.to(device), xn.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(xc, xn)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    tr_loss = total_loss / len(train_loader)
    tr_acc  = correct / total
    train_losses.append(tr_loss)
    train_accs.append(tr_acc)

    # ---- validate ----
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xc, xn, y in val_loader:
            xc, xn, y = xc.to(device), xn.to(device), y.to(device)
            logits = model(xc, xn)
            loss = criterion(logits, y)

            val_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)

    va_loss = val_loss / len(val_loader)
    va_acc  = val_correct / val_total
    val_losses.append(va_loss)
    val_accs.append(va_acc)

    # 紀錄最佳模型（以 val loss 為準）
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_state = model.state_dict()

    print(f"[Epoch {epoch:02d}] "
          f"Train Loss {tr_loss:.4f} | Train Acc {tr_acc:.2%} | "
          f"Val Loss {va_loss:.4f} | Val Acc {va_acc:.2%}")

# 若有最佳狀態，替換成最佳權重
if best_state is not None:
    model.load_state_dict(best_state)

# ========= 5) 訓練曲線（可選） =========
plt.figure(figsize=(10,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curves")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.title("Accuracy Curves")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.show()

# ========= 6) 測試集評估 =========
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xc, xn, y in test_loader:
        xc, xn, y = xc.to(device), xn.to(device), y.to(device)
        logits = model(xc, xn)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

print("\n=== Test Set Evaluation ===")
print(classification_report(y_true, y_pred, digits=4))


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Pred {i+1}" for i in range(num_classes)],
            yticklabels=[f"True {i+1}" for i in range(num_classes)])
plt.title("Confusion Matrix (Test)"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# ========= 7) 安全輸出（分離檔：權重 + 前處理 meta） =========
import torch, joblib

# 1) 只存 state_dict（純張量；部署端可 weights_only=True 載入）
torch.save(model.state_dict(), "model_weights.pt")
print("Saved model weights -> model_weights.pt")

# 2) 存前處理 meta（由 split.py 產生的 preprocess_meta.pkl 已讀入為 meta）
meta_safe = {
    "model_type": "ft",                 # 明確標示使用 FTTransformer
    "categories": categories,
    "num_continuous": num_continuous,
    "num_classes": num_classes,
    "categ_cols": categ_cols,
    "cont_cols": cont_cols,
    "label_col": label_col,
    "scaler_type": meta.get("scaler_type", "standard"),
    "scaler_mean": meta.get("scaler_mean"),
    "scaler_scale": meta.get("scaler_scale"),
    "cat_maps": meta.get("cat_maps"),
}
joblib.dump(meta_safe, "preprocess_meta.pkl")
print("Saved preprocess meta -> preprocess_meta.pkl")

# ========= 8) 推論工具（部署端可直接引用） =========

def load_model_for_infer(weights_path="model_weights.pt",
                         meta_path="preprocess_meta.pkl",
                         model_cls=FTTransformer,
                         device="cpu"):
    # 安全讀取：只載張量權重（未來 PyTorch 預設亦為 True）
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        # 舊版 PyTorch 沒有 weights_only 參數時回退
        state_dict = torch.load(weights_path, map_location=device)

    meta_local = joblib.load(meta_path)
    assert meta_local.get("model_type") == "ft", "此 loader 僅支援 FTTransformer"

    m = model_cls(
        categories=tuple(meta_local["categories"]),
        num_continuous=int(meta_local["num_continuous"]),
        dim=32, depth=3, heads=4, dim_out=int(meta_local["num_classes"])
    )
    m.load_state_dict(state_dict, strict=True)
    m.eval()
    return m, meta_local

def preprocess_one(sample_dict, meta_local):
    # 類別：依 cat_maps 編碼；未知值→0
    xc = []
    for c in meta_local["categ_cols"]:
        mapping = meta_local["cat_maps"][c]
        xc.append(mapping.get(sample_dict.get(c), 0))
    xc = torch.tensor(xc, dtype=torch.long).unsqueeze(0)

    # 連續：以訓練集 scaler 參數做 Z-score
    import numpy as np
    mu = meta_local.get("scaler_mean")
    sc = meta_local.get("scaler_scale")
    xn = np.array([float(sample_dict[k]) for k in meta_local["cont_cols"]], dtype=np.float32)
    if (mu is not None) and (sc is not None):
        xn = (xn - np.array(mu, dtype=np.float32)) / np.array(sc, dtype=np.float32)
    xn = torch.tensor(xn, dtype=torch.float32).unsqueeze(0)
    return xc, xn

# ---- 範例 ----
# model_loaded, meta_ckpt = load_model_for_infer("ft_squat_recommender.pt")
# sample = {
#   "sex":"Male","level":"Intermediate","RM_range":"70-80%RM","weekly_sets_range":"10-12",
#   "quality_score_range":"70-79 points","frequency":"3 days/week",
#   "fatigueL_desc":"Moderate","fatigueT_desc":"Moderate","labelT":"Pred 3",
#   "target_weight":100.0,"current_weight":90.0,"target_ratio":1.11,"target_weeks":12,"week":5,"recommend_weight":92.5
# }
# xc, xn = preprocess_one(sample, meta_ckpt)
# with torch.no_grad():
#     logits = model_loaded(xc, xn)
#     pred = torch.argmax(logits, dim=1).item()
# print("Pred level:", pred + 1)
