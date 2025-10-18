import torch
import json
import os
from models import GenerativeRecSysModel
import pandas as pd

def run_inference():
    print("--- Running Inference Test ---")

    # --- 1. 配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'best_model.pth')
    CATALOG_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'item_catalog.pt')
    INPUT_FILE_PATH = os.path.join(PROJECT_ROOT, 'test_user_history.txt')
    OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'inference_results.txt')
    TOP_K = 5
    MAX_SEQ_LEN = 50

    # --- 2. 加载模型和商品图谱 ---
    print(f"Loading best model from: {MODEL_PATH}")
    model = GenerativeRecSysModel(device=DEVICE).to(DEVICE)
    
    catalog = torch.load(CATALOG_PATH)
    all_item_ids = catalog['item_ids']
    all_item_embeds = catalog['embeddings'].to(DEVICE)
    
    # 【--- 错误修正处 (与 precompute.py 相同) ---】
    # 1. 先创建模型的嵌入层
    model.create_item_embeddings(all_item_ids)
    
    # 2. 再加载权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    # 【--- 修正结束 ---】
    
    # --- 3. 读取并准备输入历史 ---
    print(f"Loading user history from: {INPUT_FILE_PATH}")
    history_items = []
    with open(INPUT_FILE_PATH, 'r') as f:
        for line in f:
            history_items.append(json.loads(line))
    
    history_ids = [item['parent_asin'] for item in history_items]
    history_texts = [item['combined_text'] for item in history_items]
    history_images = [item['image_url'] for item in history_items]
    pad_len = MAX_SEQ_LEN - len(history_ids)
    input_ids, input_texts, input_images = ['<PAD>'] * pad_len + history_ids, [''] * pad_len + history_texts, [''] * pad_len + history_images
    batch_input_ids, batch_input_texts, batch_input_images = [input_ids], [input_texts], [input_images]

    # --- 4. 执行模型推理 ---
    print("\nGenerating user interest vector...")
    with torch.no_grad():
        user_interest_vector = model(batch_input_ids, batch_input_texts, batch_input_images)

    # --- 5. 进行相似度搜索 ---
    print("Searching for top K similar items...")
    scores = torch.matmul(user_interest_vector, all_item_embeds.T).squeeze(0)
    _, top_k_indices = torch.topk(scores, k=TOP_K)
    recommended_item_ids = [all_item_ids[idx] for idx in top_k_indices.cpu().numpy()]
    
    # --- 6. 整理并保存输出 ---
    full_df = pd.read_pickle(os.path.join(PROJECT_ROOT, 'data', 'processed', 'sequential_data_sample.pkl'))
    item_df = full_df.explode('parent_asin').drop_duplicates('parent_asin').set_index('parent_asin')
    
    output_records = "--- Inference Test Results ---\n\n"
    output_records += "=== Input User History ===\n"
    for item in history_items:
        output_records += f"- {item['parent_asin']}: {item['combined_text'][:80]}...\n"
    output_records += f"\n=== Top {TOP_K} Recommendations ===\n"
    for i, item_id in enumerate(recommended_item_ids):
        try:
            item_title = item_df.loc[item_id, 'combined_text']
            output_records += f"{i+1}. {item_id}: {item_title[:80]}...\n"
        except KeyError:
            output_records += f"{i+1}. {item_id}: (Title not found)\n"
    
    with open(OUTPUT_FILE_PATH, 'w') as f:
        f.write(output_records)
            
    print("\n--- Results ---")
    print(output_records)
    print(f"Inference finished. Results saved to: {OUTPUT_FILE_PATH}")

if __name__ == '__main__':
    run_inference()