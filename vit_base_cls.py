import torch
import torch.nn as nn
import torch.nn.functional as F
# import rasterio
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_chans=3,
        embed_dim=128, 
        depth=8, 
        n_heads=8, 
        mlp_ratio=4.,
        qkv_bias=True,
        p=0.1, # 
        attn_p=0.1,
        num_classes = 10
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        # Positional embedding (trainable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.mlp_head = nn.Linear(embed_dim, 10)
        

    def forward(self, x):
        # n_samples = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads, qkv_bias, attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim, p=p)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        # n_samples : No of batches
        # n_token : no of patches divided in the image per batch.
        # 3 : This is for no of matrices which is q, k, v to segregate.
        # self.n_heads : its the no of heads per q, k, v
        # self.head_dim : the no of dim per head.


        qkv = self.qkv(x).reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        dp = (q @ k.transpose(-2, -1)) * self.scale
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2).reshape(n_samples, n_tokens, dim)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


if __name__ == '__main__':

    # Patch Embedding test
    img_size = 16
    patch_size=4
    in_chans=1
    embed_dim=4

    # Batch, Channel, Height, Width
    img = torch.rand(1,1,16,16)
    obj = PatchEmbed(img_size = 16, patch_size=4, in_chans=1, embed_dim=4)
    x = obj(img)
    assert x.shape == (1,16,4), f"Expected shape [1,16,4], but got {x.shape}"
    print(f"Patch Embedding Test Passed: {x.shape}")


    # Block Check
    # x = torch.rand(1,16,4)
    obj = Block(dim=embed_dim, n_heads=1, mlp_ratio=1, attn_p=0, p=0)
    x = obj(x)

    # Attention Check is successfull
    # x = torch.rand(1,16,4)
    # obj = Attention(dim=4, n_heads=2,qkv_bias=False, attn_p=0, proj_p=0)
    # x = obj(x)

    # MLP Check is successfull

    # VisionTransformer Check is Successfull
    x = torch.rand(1,1,16,16)
    obj = VisionTransformer(        
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim, 
        depth=1, 
        n_heads=2, 
        mlp_ratio=1,
        qkv_bias=False,
        p=0,
        attn_p=0,
        num_classes=2)
    x = obj(x)









