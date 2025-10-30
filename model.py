"""
Text-to-Image Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel


class TextEncoder(nn.Module):
    """Text encoder using CLIP"""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(TextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name)
        
    def forward(self, text):
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        )
        
        # Move to device
        device = next(self.text_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get text embeddings
        outputs = self.text_model(**inputs)
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]


class AttentionBlock(nn.Module):
    """Attention mechanism to fuse text and image features"""
    def __init__(self, embed_dim):
        super(AttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, image_features, text_features):
        # image_features: [batch, channels, height, width]
        # text_features: [batch, seq_len, embed_dim]
        
        batch, channels, height, width = image_features.shape
        
        # Reshape image features to [batch, height*width, channels]
        image_flat = image_features.view(batch, channels, -1).permute(0, 2, 1)
        
        # Project features
        queries = self.query_proj(image_flat)  # [batch, height*width, embed_dim]
        keys = self.key_proj(text_features)    # [batch, seq_len, embed_dim]
        values = self.value_proj(text_features) # [batch, seq_len, embed_dim]
        
        # Compute attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # [batch, height*width, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_values = torch.bmm(attention_weights, values)  # [batch, height*width, embed_dim]
        
        # Reshape back to image format
        attended_values = attended_values.permute(0, 2, 1).view(batch, -1, height, width)
        
        # Project output
        output = self.out_proj(attended_values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return output


class TextToImageModel(nn.Module):
    """Simple text-to-image generation model"""
    def __init__(self, img_size=64, channels=3, text_embed_dim=512):
        super(TextToImageModel, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.text_embed_dim = text_embed_dim
        
        # Text encoder
        self.text_encoder = TextEncoder()
        
        # Image generator (simplified U-Net like structure)
        self.init_size = img_size // 8  # Initial size before upsampling
        self.l1 = nn.Sequential(
            nn.Linear(text_embed_dim, 128 * self.init_size ** 2)
        )
        
        # Upsampling layers
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        # Attention mechanism
        self.attention = AttentionBlock(text_embed_dim)
        
    def forward(self, text):
        # Encode text
        text_embeddings = self.text_encoder(text)  # [batch, seq_len, embed_dim]
        
        # Use the first token embedding (CLS token equivalent)
        text_features = text_embeddings[:, 0, :]  # [batch, embed_dim]
        
        # Generate initial image representation
        out = self.l1(text_features)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        
        # Apply attention with full text embeddings
        out = self.attention(out, text_embeddings)
        
        # Generate final image
        img = self.conv_blocks(out)
        return img