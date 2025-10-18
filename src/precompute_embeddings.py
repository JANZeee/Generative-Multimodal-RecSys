import torch
import pandas as pd
from tqdm import tqdm
import os
from models import GenerativeRecSysModel

def precompute():
    print("--- Starting Item Embedding Pre-computation ---")
    
    # --- 配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'sequential_data_sample.pkl')
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'best_model.pth')
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'item_catalog.pt')
    
    print(f"Using device: {DEVICE}")
    print(f"Loading best model from: {MODEL_PATH}")

    # --- 加载模型 ---
    model = GenerativeRecSysModel(device=DEVICE).to(DEVICE)
    
    # 【--- 错误修正处 ---】
    # 1. 先准备好所有唯一的商品ID
    full_df = pd.read_pickle(DATA_PATH)
    all_unique_item_ids = list(full_df['parent_asin'].explode().unique())
    
    # 2. 先创建模型的嵌入层，使其结构完整
    model.create_item_embeddings(all_unique_item_ids)
    
    # 3. 现在再加载权重，就不会有 key 不匹配的问题了
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    # 【--- 修正结束 ---】

    # --- 准备所有唯一的商品元数据 ---
    item_df = full_df.explode('parent_asin').explode('combined_text').explode('image_url').drop_duplicates('parent_asin')
    item_meta_map = item_df.set_index('parent_asin').to_dict()
    all_item_texts = [item_meta_map['combined_text'].get(item_id, "") for item_id in all_unique_item_ids]
    all_item_images = [item_meta_map['image_url'].get(item_id, "") for item_id in all_unique_item_ids]

    # --- 分批计算所有商品的嵌入 ---
    all_item_embeds = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(all_unique_item_ids), batch_size), desc="Encoding all items"):
            batch_texts = all_item_texts[i:i+batch_size]
            batch_images = all_item_images[i:i+batch_size]
            batch_embeds = model.encode_items(batch_texts, batch_images)
            all_item_embeds.append(batch_embeds)
    
    # --- 保存结果 ---
    final_embeddings = torch.cat(all_item_embeds, dim=0)
    catalog = {
        'item_ids': all_unique_item_ids,
        'embeddings': final_embeddings
    }
    torch.save(catalog, OUTPUT_PATH)
    
    print(f"\nPre-computation finished!")
    print(f"Item catalog with {final_embeddings.shape[0]} embeddings saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    precompute()