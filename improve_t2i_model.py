"""
Improved Text-to-Image Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel


class TextEncoder(nn.Module):
    """Text encoder using CLIP with device management"""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(TextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        # 冻结CLIP文本编码器参数（如需微调可删除此行）
        for param in self.text_model.parameters():
            param.requires_grad = False
        
    def forward(self, text, device):
        # 显式指定设备，避免设备不一致问题
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(device)
        
        # 获取文本特征（保留序列信息）
        outputs = self.text_model(**inputs)
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]


class AttentionBlock(nn.Module):
    """Improved cross-attention block with residual connection"""
    def __init__(self, image_channels, text_embed_dim):
        super(AttentionBlock, self).__init__()
        self.image_channels = image_channels
        self.text_embed_dim = text_embed_dim
        
        # 图像特征投影（解决维度匹配问题）
        self.image_proj = nn.Conv2d(image_channels, text_embed_dim, kernel_size=1)
        # 注意力投影层
        self.query_proj = nn.Linear(text_embed_dim, text_embed_dim)
        self.key_proj = nn.Linear(text_embed_dim, text_embed_dim)
        self.value_proj = nn.Linear(text_embed_dim, text_embed_dim)
        self.out_proj = nn.Linear(text_embed_dim, image_channels)
        # 层归一化
        self.norm = nn.BatchNorm2d(image_channels)
        
    def forward(self, image_features, text_features):
        # image_features: [batch, image_channels, height, width]
        # text_features: [batch, seq_len, text_embed_dim]
        
        batch, channels, height, width = image_features.shape
        spatial_size = height * width
        
        # 图像特征投影到文本嵌入维度
        image_proj = self.image_proj(image_features)  # [batch, text_embed_dim, h, w]
        image_flat = image_proj.view(batch, self.text_embed_dim, spatial_size).permute(0, 2, 1)  # [batch, h*w, text_embed_dim]
        
        # 注意力计算
        queries = self.query_proj(image_flat)  # [batch, h*w, text_embed_dim]
        keys = self.key_proj(text_features)    # [batch, seq_len, text_embed_dim]
        values = self.value_proj(text_features)  # [batch, seq_len, text_embed_dim]
        
        # 缩放点积注意力（防止梯度消失）
        scale = self.text_embed_dim **0.5
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / scale  # [batch, h*w, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力并重塑
        attended_values = torch.bmm(attention_weights, values)  # [batch, h*w, text_embed_dim]
        attended_values = attended_values.permute(0, 2, 1).view(batch, self.text_embed_dim, height, width)  # [batch, text_embed_dim, h, w]
        
        # 投影回图像通道并添加残差连接
        output = self.out_proj(attended_values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [batch, image_channels, h, w]
        output = self.norm(output + image_features)  # 残差连接+层归一化
        
        return output


class ResidualBlock(nn.Module):
    """Residual block for stable training"""
    def __init__(self, in_channels, out_channels, upsample=False):
        super(ResidualBlock, self).__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        if upsample:
            self.upsampler = nn.Upsample(scale_factor=2)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        residual = x
        if self.upsample:
            x = self.upsampler(x)
            residual = self.upsampler(residual)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        residual = self.shortcut(residual)
        x = x + residual
        x = self.relu(x)
        return x


class TextToImageModel(nn.Module):
    """Improved text-to-image generation model"""
    def __init__(self, img_size=64, channels=3, text_embed_dim=512):
        super(TextToImageModel, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.text_embed_dim = text_embed_dim
        
        # 文本编码器
        self.text_encoder = TextEncoder()
        
        # 初始尺寸计算
        self.init_size = img_size // 8  # 64 → 8x8
        self.l1 = nn.Sequential(
            nn.Linear(text_embed_dim, 128 * self.init_size** 2),
            nn.ReLU()  # 更稳定的激活函数
        )
        
        # 改进的生成器结构（带残差连接）
        self.conv_blocks = nn.Sequential(
            # 从128通道开始（8x8）
            ResidualBlock(128, 128),
            # 上采样到16x16
            ResidualBlock(128, 128, upsample=True),
            # 上采样到32x32
            ResidualBlock(128, 64, upsample=True),
            # 上采样到64x64
            ResidualBlock(64, 64, upsample=True),
            # 输出层
            nn.Conv2d(64, channels, 3, padding=1),
            nn.Tanh()
        )
        
        # 注意力机制（与当前图像通道匹配）
        self.attention1 = AttentionBlock(128, text_embed_dim)  # 8x8时使用
        self.attention2 = AttentionBlock(64, text_embed_dim)   # 32x32时使用
        
    def forward(self, text):
        # 获取设备（从输入文本推断或使用模型默认设备）
        device = next(self.parameters()).device
        
        # 编码文本（保留完整序列特征）
        text_embeddings = self.text_encoder(text, device)  # [batch, seq_len, text_embed_dim]
        
        # 使用CLS token初始化图像特征
        cls_feat = text_embeddings[:, 0, :]  # [batch, text_embed_dim]
        out = self.l1(cls_feat)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)  # [batch, 128, 8, 8]
        
        # 第一次注意力融合（8x8特征图）
        out = self.attention1(out, text_embeddings)
        
        # 分阶段处理卷积块，插入第二次注意力
        for i, block in enumerate(self.conv_blocks):
            out = block(out)
            # 在32x32尺寸时应用第二次注意力
            if i == 2:  # 经过两次上采样后达到32x32
                out = self.attention2(out, text_embeddings)
        
        return out