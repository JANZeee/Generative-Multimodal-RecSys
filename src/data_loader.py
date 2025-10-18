import torch
from torch.utils.data import Dataset
import pandas as pd
import random

class RecSysDataset(Dataset):
    """
    自定义PyTorch数据集，用于加载序列推荐数据。
    每一条数据代表一个用户。
    """
    def __init__(self, data_path, max_seq_len=50):
        """
        初始化函数。
        
        Args:
            data_path (str): 已处理好的序列数据文件路径 (.pkl)。
            max_seq_len (int): 序列的最大长度。超过的会被截断，不足的会被填充。
        """
        print(f"Loading data from: {data_path}")
        self.df = pd.read_pickle(data_path)
        self.max_seq_len = max_seq_len
        print(f"Data loaded. Total users: {len(self.df)}")

    def __len__(self):
        """返回数据集中的用户总数。"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个用户的训练样本。
        
        返回一个字典，包含：
        - input_item_ids: 模型的输入商品ID序列
        - input_texts: 模型的输入文本序列
        - input_images: 模型的输入图片URL序列
        - target_item_id: 模型需要预测的目标商品ID
        - target_text: 模型需要预测的目标文本
        - target_image: 模型需要预测的目标图片URL
        """
        # 获取对应用户的完整序列
        user_sequence = self.df.iloc[idx]
        
        item_ids = user_sequence['parent_asin']
        texts = user_sequence['combined_text']
        images = user_sequence['image_url']
        
        # 将序列划分为输入(input)和目标(target)
        # 输入是除了最后一个item的所有历史记录
        # 目标是最后一个item
        input_item_ids = item_ids[:-1]
        input_texts = texts[:-1]
        input_images = images[:-1]
        
        target_item_id = item_ids[-1]
        target_text = texts[-1]
        target_image = images[-1]
        
        # 对输入序列进行填充(padding)或截断(truncation)
        seq_len = len(input_item_ids)
        
        # 定义填充用的元素
        pad_item_id = '<PAD>'
        pad_text = ''
        pad_image = ''
        
        if seq_len < self.max_seq_len:
            # 如果序列长度小于最大长度，进行填充
            pad_len = self.max_seq_len - seq_len
            input_item_ids = [pad_item_id] * pad_len + input_item_ids
            input_texts = [pad_text] * pad_len + input_texts
            input_images = [pad_image] * pad_len + input_images
        else:
            # 如果序列长度大于最大长度，进行截断（只保留最新的记录）
            input_item_ids = input_item_ids[-self.max_seq_len:]
            input_texts = input_texts[-self.max_seq_len:]
            input_images = input_images[-self.max_seq_len:]
            
        return {
            "input_item_ids": input_item_ids,
            "input_texts": input_texts,
            "input_images": input_images,
            "target_item_id": target_item_id,
            "target_text": target_text,
            "target_image": target_image
        }

# ---- 你可以在 notebook 中用以下代码测试这个 Dataset 是否工作正常 ----
# from data_loader import RecSysDataset
#
# DATA_PATH = '../data/processed/sequential_data_sample.pkl'
# dataset = RecSysDataset(data_path=DATA_PATH)
# 
# # 获取第一个用户的数据看看
# sample_user_data = dataset[0] 
# print("\n--- Sample User Data ---")
# for key, value in sample_user_data.items():
#     print(f"{key}: {value}\n")