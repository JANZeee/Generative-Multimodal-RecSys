import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 确保可以从 src 目录导入我们自己的模块
from data_loader import RecSysDataset
from models import GenerativeRecSysModel

# ================================================================================= #
#  1. InfoNCE 损失函数 (不变)
# ================================================================================= #
def info_nce_loss(user_vectors, positive_item_vectors, temperature=0.07):
    # ... (这部分代码是正确的，保持不变) ...
    all_vectors = torch.cat([user_vectors, positive_item_vectors], dim=0)
    logits_matrix = torch.matmul(all_vectors, all_vectors.T) / temperature
    batch_size = user_vectors.shape[0]
    l_pos = torch.diag(logits_matrix, batch_size)
    r_pos = torch.diag(logits_matrix, -batch_size)
    positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=user_vectors.device)
    negatives = logits_matrix[~mask].view(2 * batch_size, -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=user_vectors.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

# ================================================================================= #
#  2. 训练与评估的核心函数 (不变)
# ================================================================================= #
def train_one_epoch(model, dataloader, optimizer, device):
    # ... (内容不变) ...
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids, input_texts, input_images = batch['input_item_ids'], batch['input_texts'], batch['input_images']
        target_texts, target_images = batch['target_text'], batch['target_image']
        optimizer.zero_grad()
        user_interest_vectors = model(input_ids, input_texts, input_images)
        with torch.no_grad():
            positive_item_vectors = model.encode_items(target_texts, target_images)
        loss = info_nce_loss(user_interest_vectors, positive_item_vectors)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, all_item_ids, all_item_embeds, device, k=10):
    # ... (内容不变) ...
    model.eval()
    total_hr, total_ndcg, num_samples = 0, 0, 0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    for batch in progress_bar:
        user_interest_vectors = model(batch['input_item_ids'], batch['input_texts'], batch['input_images'])
        scores = torch.matmul(user_interest_vectors, all_item_embeds.T)
        for i in range(user_interest_vectors.shape[0]):
            target_item_id = batch['target_item_id'][i]
            _, top_k_indices = torch.topk(scores[i], k=len(all_item_ids))
            ranked_item_ids = [all_item_ids[idx] for idx in top_k_indices.cpu().numpy()]
            try:
                rank = ranked_item_ids.index(target_item_id) + 1
                if rank <= k:
                    total_hr += 1
                    total_ndcg += 1 / np.log2(rank + 1)
            except ValueError: pass
        num_samples += user_interest_vectors.shape[0]
    return total_hr / num_samples, total_ndcg / num_samples

# ================================================================================= #
#  3. 训练主流程
# ================================================================================= #
def main():
    # --- 1. 超参数与配置 (不变) ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
    PLOT_SAVE_DIR = os.path.join(OUTPUT_DIR, 'plots')
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
    PLOT_SAVE_PATH = os.path.join(PLOT_SAVE_DIR, 'training_curves.png')
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sequential_data_sample.pkl')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
    BATCH_SIZE, LEARNING_RATE, EPOCHS, MAX_SEQ_LEN, EVAL_K = 64, 1e-4, 10, 50, 10
    print(f"Using device: {DEVICE}\nModel will be saved to: {MODEL_SAVE_PATH}\nPlots will be saved to: {PLOT_SAVE_PATH}")

    # --- 2. 数据集准备与划分 ---
    full_df = pd.read_pickle(DATA_PATH)
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    def new_dataset_init(self, df, max_len):
        self.df = df.reset_index(drop=True)
        self.max_seq_len = max_len
    RecSysDataset.__init__ = new_dataset_init
    train_dataset = RecSysDataset(train_df, MAX_SEQ_LEN)
    val_dataset = RecSysDataset(val_df, MAX_SEQ_LEN)
    
    # 【--- 错误修正处 ---】
    # 使用一个简单、正确、健壮的 collate_fn，它只做一件事：把数据按key分组。
    def correct_collate_fn(batch):
        """
        输入: [{'key1': valA1, 'key2': valB1}, {'key1': valA2, 'key2': valB2}]
        输出: {'key1': [valA1, valA2], 'key2': [valB1, valB2]}
        """
        keys = batch[0].keys()
        return {key: [d[key] for d in batch] for key in keys}
    # 【--- 修正结束 ---】

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=correct_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=correct_collate_fn)

    # --- 3. 模型, 优化器, 历史记录 初始化 (不变) ---
    model = GenerativeRecSysModel(device=DEVICE).to(DEVICE)
    all_unique_item_ids = list(full_df['parent_asin'].explode().unique())
    model.create_item_embeddings(all_unique_item_ids)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = {'train_loss': [], 'val_hr': [], 'val_ndcg': []}
    best_ndcg = 0.0

    # --- 4. 预计算所有物品的嵌入 (不变) ---
    print("Pre-computing all item embeddings for evaluation...")
    item_df = full_df.explode('parent_asin').explode('combined_text').explode('image_url').drop_duplicates('parent_asin')
    item_meta_map = item_df.set_index('parent_asin').to_dict()
    all_item_texts = [item_meta_map['combined_text'].get(item_id, "") for item_id in all_unique_item_ids]
    all_item_images = [item_meta_map['image_url'].get(item_id, "") for item_id in all_unique_item_ids]
    with torch.no_grad():
        all_item_embeds = model.encode_items(all_item_texts, all_item_images)

    # --- 5. 训练主循环 (不变) ---
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, DEVICE)
        hr_at_k, ndcg_at_k = evaluate(model, val_dataloader, all_unique_item_ids, all_item_embeds, DEVICE, k=EVAL_K)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | HR@{EVAL_K}: {hr_at_k:.4f} | NDCG@{EVAL_K}: {ndcg_at_k:.4f}")
        history['train_loss'].append(avg_train_loss)
        history['val_hr'].append(hr_at_k)
        history['val_ndcg'].append(ndcg_at_k)
        if ndcg_at_k > best_ndcg:
            best_ndcg = ndcg_at_k
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved with NDCG@{EVAL_K}: {best_ndcg:.4f}")

    print("\n--- Training Finished ---")

    # --- 6. 绘制并保存训练曲线 (不变) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    epochs_range = range(1, EPOCHS + 1)
    ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Training Loss')
    ax1.set_ylabel('Loss'), ax1.set_title('Training Loss'), ax1.legend()
    ax2.plot(epochs_range, history['val_hr'], 'r-o', label=f'HR@{EVAL_K}')
    ax2.plot(epochs_range, history['val_ndcg'], 'g-o', label=f'NDCG@{EVAL_K}')
    ax2.set_xlabel('Epoch'), ax2.set_ylabel('Score'), ax2.set_title('Validation Metrics'), ax2.legend()
    plt.tight_layout(), plt.savefig(PLOT_SAVE_PATH)
    print(f"Training curves saved to {PLOT_SAVE_PATH}")

if __name__ == '__main__':
    main()