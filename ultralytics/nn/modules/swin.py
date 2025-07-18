# ultralytics/nn/modules/swin.py
import torch.nn as nn
import timm

class SwinTinyBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load Swin‑Tiny from timm, return intermediate features
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True,
                                       out_indices=(1, 2, 3))  
        # out_indices: stage1→P3, stage2→P4, stage3→P5

    def forward(self, x):
        # returns list of feature maps [P3, P4, P5]
        feats = self.model(x)
        return feats
