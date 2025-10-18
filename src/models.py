import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import warnings
import math

warnings.filterwarnings("ignore")

class MultimodalEncoder(nn.Module):
    # ... (这部分代码没有变化，保持原样) ...
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu", freeze=True):
        super().__init__()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _get_image_from_url(self, url):
        if not url or not url.startswith('http'): return None
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except:
            return None

    def forward(self, texts, image_urls):
        images = [self._get_image_from_url(url) for url in image_urls]
        placeholder_image = Image.new('RGB', (self.model.config.vision_config.image_size, self.model.config.vision_config.image_size), (255, 255, 255))
        valid_images = [img if img is not None else placeholder_image for img in images]
        inputs = self.processor(text=texts, images=valid_images, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        multimodal_embeds = image_embeds + text_embeds
        multimodal_embeds = nn.functional.normalize(multimodal_embeds, p=2, dim=-1)
        return multimodal_embeds

class PositionalEncoding(nn.Module):
    # ... (这部分代码没有变化，保持原样) ...
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class GenerativeRecSysModel(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, num_layers=3, dropout=0.1, device="cpu"):
        super().__init__()
        self.device = device
        self.multimodal_encoder = MultimodalEncoder(device=device)
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid()
        ).to(device)
        self.item_id_embedding = None
        self.output_layer = nn.Linear(embed_dim, embed_dim).to(device)

        # 【--- 错误修正处 ---】
        # 增加一个 LayerNorm 层用于稳定进入 Transformer 的输入
        self.fusion_layernorm = nn.LayerNorm(embed_dim).to(device)
        # 【--- 修正结束 ---】

    def create_item_embeddings(self, all_item_ids):
        self.unique_item_ids = list(all_item_ids)
        self.item_id_to_idx = {item_id: i for i, item_id in enumerate(self.unique_item_ids)}
        self.item_id_embedding = nn.Embedding(len(self.unique_item_ids) + 1, self.multimodal_encoder.model.config.projection_dim).to(self.device)
        self.pad_idx = len(self.unique_item_ids)

    def encode_items(self, item_texts, item_image_urls):
        return self.multimodal_encoder(item_texts, item_image_urls)

    def forward(self, input_item_ids, input_texts, input_images):
        batch_size, seq_len = len(input_item_ids), len(input_item_ids[0])
        flat_texts = [text for user_seq in input_texts for text in user_seq]
        flat_image_urls = [url for user_seq in input_images for url in user_seq]
        multimodal_embeds = self.encode_items(flat_texts, flat_image_urls)
        multimodal_embeds = multimodal_embeds.view(batch_size, seq_len, -1)
        
        if self.item_id_embedding is None:
            raise RuntimeError("Item ID embedding layer is not initialized. Call `create_item_embeddings` first.")
        
        item_indices = torch.tensor(
            [[self.item_id_to_idx.get(item_id, self.pad_idx) for item_id in seq] for seq in input_item_ids],
            dtype=torch.long, device=self.device
        )
        id_embeds = self.item_id_embedding(item_indices)
        
        gate_weights = self.fusion_gate(multimodal_embeds)
        fused_embeds = (1 - gate_weights) * multimodal_embeds + gate_weights * id_embeds
        
        # 【--- 错误修正处 ---】
        # 在进入 Transformer 前应用 LayerNorm
        fused_embeds_norm = self.fusion_layernorm(fused_embeds)
        # 【--- 修正结束 ---】

        fused_embeds_pos = self.pos_encoder(fused_embeds_norm) # 使用归一化后的嵌入
        
        padding_mask = (item_indices == self.pad_idx)
        transformer_output = self.transformer_encoder(fused_embeds_pos, src_key_padding_mask=padding_mask)
        
        user_interest_vector = transformer_output[:, -1, :]
        return user_interest_vector