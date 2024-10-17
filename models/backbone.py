# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from sam2.build_sam import build_sam2

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
            self.conv_strides_to_apply = [2,1,1]
            self.k_size_strides_to_apply = [3,1,1]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
            self.conv_strides_to_apply = []
            self.k_size_strides_to_apply = []
            
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = []
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class ModifiedImageEncoderViT(nn.Module):
    def __init__(self, original_model, return_interm_layers):
        super(ModifiedImageEncoderViT, self).__init__()

        # Extract all layers up to but not including 'neck'
        self.patch_embed = original_model.trunk.patch_embed
        self.blocks = original_model.trunk.blocks
        self.pos_embed = original_model.trunk.pos_embed
        self.pos_embed_window = original_model.trunk.pos_embed_window
        self.return_interm_layers = return_interm_layers

        stages = (2, 3, 16, 3) # extracted from https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
        depth = sum(stages)
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]

    def _get_pos_embed(self, hw) -> torch.Tensor:
        """ 
        Extracted from Hiera
        https://github.com/facebookresearch/segment-anything-2/blob/main/sam2/modeling/backbones/hieradet.py#L265
        """
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x):
        # Apply patch embedding
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            # Add pos embed
            x = x + self._get_pos_embed(x.shape[1:3])

        # # Pass through blocks
        # for block in self.blocks:
        #     x = block(x)
        # return x.permute(0, 3, 1, 2)

        # For returning intermediate feature layers 
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
                
        return outputs

class SAM2_Backbone(nn.Module):
    def __init__(self, checkpoint, model_cfg, return_interm_layers=False):
        super().__init__()
        self.encoder = build_sam2(model_cfg, checkpoint).image_encoder
        # self.encoder = ModifiedImageEncoderViT(self.encoder, return_interm_layers)

        # Freeze encoder
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad_(False)
        if return_interm_layers:
            self.num_channels = [256,256,256]
            self.strides = [4,8,16]
            self.conv_strides_to_apply = [4,2,1]
            self.k_size_strides_to_apply = [4,1,1]
        else:
            self.num_channels = [768]
            self.strides = [16]

    def forward(self, tensor_list: NestedTensor):
        """
            'vision_features': features 
            'vision_pos_enc' (3 items): pos encoding 
            'backbone_fpn' (3 items): features at different scales . The last item is the same as 'vision_features'
                (Pdb) xs['backbone_fpn'][0].shape
                torch.Size([1, 256, 160, 144])
                (Pdb) xs['backbone_fpn'][1].shape
                torch.Size([1, 256, 80, 72])
                (Pdb) xs['backbone_fpn'][2].shape
                torch.Size([1, 256, 40, 36])
        """
        # x = self.encoder(tensor_list.tensors)
        # # print(x['vision_features'].shape) # torch.Size([1, 256, 64, 64])
        # m = tensor_list.mask  
        # assert m is not None
        # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        # out = NestedTensor(x, mask)
        # return [out]
    
        xs = self.encoder(tensor_list.tensors)
        out = []
        for x in xs['backbone_fpn']:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        return out
    

def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    if "resnet" in args.backbone:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    else: # "sam2" == args.backbone:
        backbone = SAM2_Backbone(args.sam2_checkpoint, args.sam2_model_cfg, return_interm_layers)
    return backbone
