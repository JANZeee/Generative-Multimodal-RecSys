# Generative Multimodal Recommendation System
### 生成式多模态推荐系统

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-blue)

[English](#-english-version) | [中文版](#-中文版)

---

## 🇬🇧 English Version

### 1. Overview
This project is a generative multimodal recommendation system that predicts the next item a user is likely to purchase. It leverages a user's historical interaction sequence, incorporating both **textual (descriptions, reviews)** and **visual (product images)** information.

-   **Task:** Next-Item Prediction / Sequential Recommendation
-   **Dataset:** [Amazon Reviews' 23](https://amazon-reviews-2023.github.io/) ("Handmade" & "Health/Personal Care" categories).

### 2. Model Architecture
Our model (`GenerativeRecSysModel`) is an end-to-end network composed of three key stages:
1.  **Multimodal Encoder:** Uses a pre-trained **CLIP** model to encode item images and texts into unified semantic vectors.
2.  **Dynamic Feature Fusion:** A **gating mechanism** adaptively fuses the semantic vectors with traditional ID embeddings.
3.  **Sequential Modeling:** A **Transformer Encoder** processes the sequence of fused vectors to capture user's dynamic interests and generate a final user interest vector for prediction.
4.  **Training:** Optimized with an **InfoNCE** contrastive loss function.

### 3. Quick Start
1.  **Setup Environment:**
    ```bash
    git clone https://github.com/JANZeee/Generative-Multimodal-RecSys.git
    cd Generative-Multimodal-RecSys
    conda create -n recsys_proj python=3.9
    conda activate recsys_proj
    pip install -r requirements.txt
    ```

2.  **Prepare Data:**
    -   Download the `reviews` and `metadata` files for "Handmade" and "Health_and_Personal_Care" from the dataset's official website.
    -   Place the four `.jsonl.gz` files into the `data/raw/` directory.
    -   Run all cells in `notebooks/preprocessing.ipynb` to generate the processed data.

3.  **Train the Model:**
    ```bash
    cd src/
    python train.py
    ```
    -   The best model is saved to `outputs/models/best_model.pth`.
    -   Training curves are saved to `outputs/plots/training_curves.png`.

4.  **Run Inference:**
    -   Create a `test_user_history.txt` file in the project root.
    -   Run `python precompute_embeddings.py` once to build an item catalog.
    -   Run `python inference.py` to get recommendations. Results are saved to `outputs/inference_results.txt`.

---
<br>

## 🇨🇳 中文版

### 1. 项目概述
本项目是一个生成式多模态推荐系统，旨在预测用户可能购买的下一个商品。系统深度利用了用户的历史交互序列，并融合了商品的**文本信息（描述、评论）**与**视觉信息（商品图片）**。

-   **任务类型：** 下一商品预测 / 序列化推荐
-   **数据集：** [Amazon Reviews' 23](https://amazon-reviews-2023.github.io/) (专注于 "Handmade" 和 "Health/Personal Care" 两个品类).

### 2. 模型架构
我们的模型 (`GenerativeRecSysModel`) 是一个端到端的神经网络，包含三个核心阶段：
1.  **多模态编码器：** 使用预训练的 **CLIP** 模型将商品的图片和文本编码为统一的语义向量。
2.  **动态特征融合：** 采用**门控机制**自适应地融合语义向量与传统的ID嵌入。
3.  **序列行为建模：** 基于 **Transformer Encoder** 结构处理融合后的向量序列，以捕捉用户的动态兴趣，并生成最终的用户兴趣向量用于预测。
4.  **训练策略：** 使用 **InfoNCE** 对比学习损失函数进行优化。

### 3. 快速开始
1.  **环境配置：**
    ```bash
    git clone https://github.com/JANZeee/Generative-Multimodal-RecSys.git
    cd Generative-Multimodal-RecSys
    conda create -n recsys_proj python=3.9
    conda activate recsys_proj
    pip install -r requirements.txt
    ```

2.  **数据准备：**
    -   从数据集官网下载 "Handmade" 和 "Health_and_Personal_Care" 品类的 `reviews` 和 `metadata` 文件。
    -   将这四个 `.jsonl.gz` 文件放入 `data/raw/` 目录。
    -   运行 `notebooks/preprocessing.ipynb` 中的所有单元格，以生成处理好的数据。

3.  **模型训练：**
    ```bash
    cd src/
    python train.py
    ```
    -   最优模型将保存至 `outputs/models/best_model.pth`。
    -   训练曲线图将保存至 `outputs/plots/training_curves.png`。

4.  **执行推理：**
    -   在项目根目录创建一个 `test_user_history.txt` 文件作为输入。
    -   运行一次 `python precompute_embeddings.py` 来构建全商品索引。
    -   运行 `python inference.py` 以获取推荐结果。结果将保存至 `outputs/inference_results.txt`。
