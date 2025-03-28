import inspect
import math
import os
import sys

import torch
from torch import nn

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)

import torch.nn.functional as Fun
from model_zoo.feature_extrc.utils import trunc_normal_
from model_zoo.feature_extrc.deit_features import deit_base_patch_features


class ViT_VaR(nn.Module):
    """VLFAT with learnable positional encoding"""

    def __init__(self, model_type, pretrained, num_classes, dim=768, n_frames=20, logger=None,
                 interpolation_type='nearest', depth=12, heads=3, reserve_token_nums=81, discard_ratio=0.9, head_fusion='mean'):
        super().__init__()
        self.logger = logger
        self.model_type, self.pretrained, self.num_classes, self.n_frames = model_type, pretrained, num_classes, n_frames
        assert interpolation_type in ['nearest', 'linear']
        self.interpolation_type = interpolation_type

        self.feature_extractor_spacial = deit_base_patch_features(pretrained=self.pretrained, nb_classes=0, drop=0.0,
                                                                  drop_path=0.1)
        self.feature_extractor_spacial.reserve_token_nums = reserve_token_nums

        self.feature_extractor_spacial.discard_ratio = discard_ratio
        self.feature_extractor_spacial.head_fusion = head_fusion

        self.logger.info(f'[INFO] before feature_extractor_spacial.reserve_layers: {self.feature_extractor_spacial.reserve_layers}')
        self.logger.info(f'[INFO] before feature_extractor_spacial.reserve_token_nums: {self.feature_extractor_spacial.reserve_token_nums}')
        self.logger.info(f'[INFO] before feature_extractor_spacial.discard_ratio: {self.feature_extractor_spacial.discard_ratio}')
        self.logger.info(f'[INFO] before feature_extractor_spacial.head_fusion: {self.feature_extractor_spacial.head_fusion}')

        self.feature_extractor_spacial.filter_tokens = False
        from model_zoo.ViViTs.models import Transformer
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        num_patches = self.n_frames + 1
        self.temporal_pos_encodings = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.feature_extractor_temporal = Transformer(dim, depth=depth, heads=heads, dim_head=192, mlp_dim=4 * dim)
        self.pos_drop = nn.Dropout(p=0.5)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        trunc_normal_(self.temporal_pos_encodings, std=.02)
        trunc_normal_(self.temporal_token, std=.02)




    def pose_embd(self, num_patch_new):
        if self.temporal_pos_encodings.shape[1] != num_patch_new:
            # self.logger.info(f' Adjusting PE for {num_patch_new}...')
            tmp = Fun.interpolate(self.temporal_pos_encodings.transpose(-2, -1),
                                  num_patch_new,
                                  mode=self.interpolation_type).transpose(-2, -1)
            self.temporal_pos_encodings = nn.Parameter(tmp)

        return self.temporal_pos_encodings

    def __call__(self, x, return_features=False):
        # x shape B, T, C, W, H
        b, T, C, W, H = x.shape
        features = torch.stack([self.feature_extractor_spacial(_) for _ in x])
        cls_tokens = self.temporal_token.expand(b, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)
        features = features + self.pose_embd(
            num_patch_new=features.shape[1])  # features embedding already include the temporal cls token
        features = self.pos_drop(features)
        features = self.feature_extractor_temporal(features)
        cls_token_temporal = features[:, 0]
        if return_features:
            return self.cls_head(cls_token_temporal), cls_token_temporal
        return self.cls_head(cls_token_temporal)


class ViT_baseline(nn.Module):
    """FAT with learnable positional encoding"""

    def __init__(self, model_type, pretrained, num_classes, dim=768, n_frames=20, noPE=False, logger=None,
                 reserve_token_nums=81, discard_ratio=0.9, head_fusion='mean'):
        super().__init__()

        self.model_type, self.pretrained, self.num_classes, self.n_frames = model_type, pretrained, num_classes, n_frames
        self.noPE = noPE
        self.logger = logger

        self.feature_extractor_spacial = deit_base_patch_features(pretrained=self.pretrained,
                                                                  nb_classes=0,
                                                                  drop=0.0,
                                                                  drop_path=0.1)
        self.feature_extractor_spacial.reserve_token_nums = reserve_token_nums
        self.feature_extractor_spacial.discard_ratio = discard_ratio
        self.feature_extractor_spacial.head_fusion = head_fusion

        self.logger.info(f'[INFO] before feature_extractor_spacial.reserve_layers: {self.feature_extractor_spacial.reserve_layers}')
        self.logger.info(f'[INFO] before feature_extractor_spacial.reserve_token_nums: {self.feature_extractor_spacial.reserve_token_nums}')
        self.logger.info(f'[INFO] before feature_extractor_spacial.discard_ratio: {self.feature_extractor_spacial.discard_ratio}')
        self.logger.info(f'[INFO] before feature_extractor_spacial.head_fusion: {self.feature_extractor_spacial.head_fusion}')

        from model_zoo.ViViTs.models import Transformer
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        """Part for positional encoding"""
        num_patches = self.n_frames
        self.temporal_pos_encodings = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=0.5)
        trunc_normal_(self.temporal_pos_encodings, std=.02)
        """"""
        self.feature_extractor_temporal = Transformer(dim, depth=12, heads=3, dim_head=192, mlp_dim=4 * dim)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        trunc_normal_(self.temporal_token, std=.02)

    def __call__(self, x, return_features=False):
        # x shape B, T, C, W, H
        b, T, C, W, H = x.shape
        features = torch.stack([self.feature_extractor_spacial(_) for _ in x])
        cls_tokens = self.temporal_token.expand(b, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)

        "Ignore/Include PE"
        if not self.noPE:
            features = features + self.temporal_pos_encodings
            features = self.pos_drop(features)

        features = self.feature_extractor_temporal(features)

        cls_token_temporal = features[:, 0]
        if return_features:
            return self.cls_head(cls_token_temporal), cls_token_temporal
        return self.cls_head(cls_token_temporal)

class ViT_baseline_thickness(nn.Module):
    """FAT with learnable positional encoding"""

    def __init__(self, model_type, pretrained, num_classes, dim=768, n_frames=20, noPE=False, logger=None,
                 reserve_token_nums=81, discard_ratio=0.9, head_fusion='mean',interpolation_type='nearest'):
        super().__init__()

        self.model_type, self.pretrained, self.num_classes, self.n_frames = model_type, pretrained, num_classes, n_frames
        self.noPE = noPE
        self.logger = logger
        self.dim = dim

        self.feature_extractor_spacial = deit_base_patch_features(pretrained=self.pretrained,
                                                                  nb_classes=0,
                                                                  drop=0.0,
                                                                  drop_path=0.1)
        self.feature_extractor_spacial.reserve_token_nums = reserve_token_nums
        self.feature_extractor_spacial.discard_ratio = discard_ratio
        self.feature_extractor_spacial.head_fusion = head_fusion
        self.interpolation_type = interpolation_type

        # self.logger.info(f'[INFO] before feature_extractor_spacial.reserve_layers: {self.feature_extractor_spacial.reserve_layers}')
        # self.logger.info(f'[INFO] before feature_extractor_spacial.reserve_token_nums: {self.feature_extractor_spacial.reserve_token_nums}')
        # self.logger.info(f'[INFO] before feature_extractor_spacial.discard_ratio: {self.feature_extractor_spacial.discard_ratio}')
        # self.logger.info(f'[INFO] before feature_extractor_spacial.head_fusion: {self.feature_extractor_spacial.head_fusion}')

        from model_zoo.ViViTs.models import Transformer
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        """Part for positional encoding"""
        num_patches = self.n_frames
        self.temporal_pos_encodings = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(p=0.5)
        trunc_normal_(self.temporal_pos_encodings, std=.02)
        """"""
        self.feature_extractor_temporal = Transformer(dim, depth=12, heads=3, dim_head=192, mlp_dim=4 * dim)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        trunc_normal_(self.temporal_token, std=.02)
    def pose_embd(self, num_patch_new):
        if self.temporal_pos_encodings.shape[1] != num_patch_new:
            # self.logger.info(f' Adjusting PE for {num_patch_new}...')
            tmp = Fun.interpolate(self.temporal_pos_encodings.transpose(-2, -1),
                                  num_patch_new,
                                  mode=self.interpolation_type).transpose(-2, -1)
            self.temporal_pos_encodings = nn.Parameter(tmp)
            #trunc_normal_(self.temporal_pos_encodings, std=.02)

        return self.temporal_pos_encodings
    def __call__(self, x, thickness, return_features=False):
        # x shape B, T, C, W, H
        b, T, C, W, H = x.shape
        features = torch.stack([self.feature_extractor_spacial(_) for _ in x])
        """Multiply thickness into features"""
        # expand thickness to match the features shape
        thickness = thickness.unsqueeze(-1)
        features = features * thickness.expand(-1, -1, self.dim)
        cls_tokens = self.temporal_token.expand(b, -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)

        # "Ignore/Include PE"
        # if not self.noPE:
        #     features = features + self.temporal_pos_encodings
        #     features = self.pos_drop(features)
        features = features + self.pose_embd(
            num_patch_new=features.shape[1])  # features embedding already include the temporal cls token
        features = self.pos_drop(features)
        features = self.feature_extractor_temporal(features)

        cls_token_temporal = features[:, 0]
        if return_features:
            return self.cls_head(cls_token_temporal), cls_token_temporal
        return self.cls_head(cls_token_temporal)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    # device = torch.device('cpu')
    model = ViT_baseline_thickness('vit_base_patch16_224', pretrained=False,  num_classes=2, dim=768,n_frames=2)

    inp = torch.ones([3, 2, 3, 224, 224])
    thickness = torch.ones([3, 2])
    features = model(inp, thickness)


    # from timm.models.vision_transformer import vit_base_patch16_224
    # model = vit_base_patch16_224()
    # input = torch.ones((1, 3, 224, 224))
    # model(input)

    # model = deit_base_patch_features()
    # model.filter_tokens  = True
    # input = torch.ones((2, 3, 224, 224))
    # featrues = model(input)
    #
    # print(featrues.shape)
