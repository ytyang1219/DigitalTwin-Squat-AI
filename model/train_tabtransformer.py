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
# from tab_transformer_pytorch import TabTransformer
from demo.modal.ft_transformer import FTTransformer



# === 1. 資料前處理 ===
def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    # # 四捨五入至
    # df["expected_weight"] = (df["expected_weight"] / 2.5).round() * 2.5
    # df["current_weight"] = (df["current_weight"] / 2.5).round() * 2.5

    # 類別特徵做 Label Encoding
    categ_cols = [
        'sex',                 # 性別
        'level',               # 訓練等級
        'RM_range',            # 訓練等級
        'weekly_sets_range',   # 一週組數區間
        'quality_score_range', # 訓練品質得分區間
        'frequency',
        'labelT',          # 本週訓練強度
        'fatigueL_desc',       # 本週疲勞
        'fatigueT_desc'       # 目標疲勞
    ]
    for col in categ_cols:
        df[col] = df[col].astype('category').cat.codes

    # label 從 0 開始
    df["label"] = df["label"] - df["label"].min()

    return df, categ_cols

# === 2. Dataset 類別 ===
class TabDataset(Dataset):
    def __init__(self, df, categ_cols, cont_cols):
        self.y = torch.tensor(df["label"].values, dtype=torch.long)
        self.x_categ = torch.tensor(df[categ_cols].values, dtype=torch.long)
        self.x_cont = torch.tensor(df[cont_cols].values, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_categ[idx], self.x_cont[idx], self.y[idx]

# === 3. 讀取資料 ===
train_df, categ_cols = preprocess_data("train.csv")
val_df, _ = preprocess_data("val.csv")
test_df, _ = preprocess_data("test.csv")

cont_cols = [                   
    'target_weight',       # 目標重量
    'current_weight',      # 目前主力訓練重量
    'target_ratio',        # 目標重量/當前重量
    'target_weeks',        # 總週期
    'week',                # 當前週次
    'recommend_weight'     # 下週建議重量
]

# 動態計算類別數量
categories = tuple([train_df[col].nunique() for col in categ_cols])
num_continuous = len(cont_cols)
num_classes = train_df["label"].nunique()

# === 4. 建立 DataLoader ===
train_loader = DataLoader(TabDataset(train_df, categ_cols, cont_cols), batch_size=32, shuffle=True)
val_loader = DataLoader(TabDataset(val_df, categ_cols, cont_cols), batch_size=32)
test_loader = DataLoader(TabDataset(test_df, categ_cols, cont_cols), batch_size=32)


# === 4. 模型初始化 ===
# model = TabTransformer(
#     categories=categories,
#     num_continuous=num_continuous,
#     dim=32,
#     depth=3,
#     heads=4,
#     dim_out=num_classes
# )
model = FTTransformer(
    categories=categories,
    num_continuous=num_continuous,
    dim=32,
    depth=3,
    heads=4,
    dim_out=num_classes
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === 5. 損失與最佳化器 ===
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === 6. 訓練準備 ===
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

start_time = time.time()
num_epochs = 20

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x_categ, x_cont, y in train_loader:
        x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x_categ, x_cont)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # === 驗證 ===
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x_categ, x_cont ,y in val_loader:
            x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
            out = model(x_categ, x_cont)
            loss = loss_fn(out, y)

            val_loss += loss.item()
            preds = torch.argmax(out, dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"[Epoch {epoch:3d}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

# === 7. 測試集評估 ===
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for x_categ, x_cont, y in test_loader:
        x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
        out = model(x_categ, x_cont)
        preds = torch.argmax(out, dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

print("\n=== Test Set Evaluation ===")
print(classification_report(y_true, y_pred, digits=4))

# === 8. 混淆矩陣 ===
conf_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Pred {i+1}' for i in range(num_classes)],
            yticklabels=[f'True {i+1}' for i in range(num_classes)])
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# === 9. 損失曲線 ===
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Val Loss', color='green')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 10. 準確率曲線 ===
plt.figure(figsize=(10, 4))
plt.plot(train_accuracies, label='Train Accuracy', color='orange')
plt.plot(val_accuracies, label='Val Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 11. 最終結果輸出 ===
end_time = time.time()
test_acc = sum([t == p for t, p in zip(y_true, y_pred)]) / len(y_true)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Total training and evaluation time: {end_time - start_time:.2f} seconds")
