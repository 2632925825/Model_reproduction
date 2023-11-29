"""
ContextCluster implementation
# --------------------------------------------------------
# Context Cluster -- Image as Set of Points, ICLR'23 Oral
# Licensed under The MIT License [see LICENSE for details]
# Written by Xu Ma (ma.xu1@northeastern.com)
# --------------------------------------------------------
"""
import os
import copy
import torch
import torch.nn as nn
import pdb
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}


class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    对输入张量的最后一个维度进行归一化
    再求余弦相似度 为什么要进行这样的操作 这样的操作对模型有什么好处 这么做的目的是什么
    有没有等价的操作 这个操作背后的数学思想是什么
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    # 矩阵相乘操作常被用于计算特征之间的相似度、注意力权重等
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """

        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center

    def forward(self, x):  # [b,c,w,h]
        # pdb.set_trace()
        # input torch.Size([32, 32, 56, 56])
        value = self.v(x)  # torch.Size([32, 96, 56, 56])
        x = self.f(x)   #  torch.Size([32, 96, 56, 56])
        # 首先对channel进行划分，引入多头机制
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)  # torch.Size([128, 24, 56, 56])
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads) # torch.Size([128, 24, 56, 56])
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            # 在将窗口细化 堆叠到batch的维度
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]] torch.Size([8192, 24, 7, 7])
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # torch.Size([B, 24, 2, 2]), we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # torch.Size([B, 4, 24]) 
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N] torch.Size([8192, 4, 49])
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True) # 获得跟中心点最相似的小窗口的相似值和索引
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D] torch.Size([8192, 49, 24])
        # aggregate step, out shape [B,M,D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                    sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D] torch.Size([8192, 4, 24])

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww) # 返回中心点的值 能不能让这些中心点之间的距离分开
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)
        # 进行复原
        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out) 
        # torch.Size([32, 32, 56, 56])
        # torch.Size([32, 64, 28, 28])
        # torch.Size([32, 196, 14, 14])
        # torch.Size([32, 320, 7, 7])
        return out


class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim, return_center=False)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep ContextClusters.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * ( block_idx + sum(layers[:index])) / (sum(layers) - 1)  # 这个drop_path 数学理论是啥？ 似乎现在都在用
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim, return_center=False
        ))
    blocks = nn.Sequential(*blocks)

    return blocks


class ContextCluster(nn.Module):
    """
    ContextCluster, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=4, in_stride=4, in_pad=0,
                 down_patch_size=2, down_stride=2, down_pad=0,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=True,
                 init_cfg=None,
                 pretrained=None,
                 # the parameters for context-cluster
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32],
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        # 第一个 嵌入层 输入的 C 维度是5（加上了位置信息）
        self.patch_embed = PointRecuder(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=5, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                 fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                 return_center=False
                                 )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PointRecuder(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)  

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        _, c, img_w, img_h = x.shape
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        x = self.patch_embed(torch.cat([x, pos], dim=1))
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat: 
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1])) # 输出的张量是 [batch, classes]
        # for image classification
        return cls_out

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import pdb
import copy

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        
        else:
            self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
    
    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class Conv(nn.Module):
    '''Normal Conv with SiLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        # pdb.set_trace()
        x_l, x_m, x_s, x_n = x #  这里需要修改一下 x_n是特征图最小的那个 需要用到插值方法 统一S
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([x_l, x_m, x_s, x_n], 1) # 在第一个channel数进行拼接 (B, C, S)
        return out

class GCNCombiner(nn.Module):

    def __init__(self, num_classes, total_num_selects, proj_size):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        # build one layer structure (with adaptive module) 构建邻接矩阵所需的维度
        num_joints = total_num_selects // 16  # 特征长度除以32取余后 这个整数作为邻接矩阵的大小  这个超参数是可以调的
        self.param_pool0 = nn.Linear(total_num_selects, num_joints)
        # 将输入特征的总长度映射到一个的固定长度的向量中
        A = torch.eye(num_joints) / 100 + 1 / 100  # 创建一个对角矩阵
        self.adj1 = nn.Parameter(copy.deepcopy(A))  # 可训练参数，表示图卷积层网络的邻接矩阵
        self.conv1 = nn.Conv1d(proj_size, proj_size, 1)
        # 表示输入特征的通道数，也就是卷积核的输入通道数，而输出通道数也为proj_size，卷积核大小为1，
        # 即只在序列的一个位置上进行卷积操作。这个卷积核的作用是对输入特征进行通道间的信息交换和整合，以增强模型的表达能力。
        self.batch_norm1 = nn.BatchNorm1d(proj_size)  # 对输入数据进行归一化 使得每个特征的均值为0 方差为1 第一个参数为输入张量的特征数

        self.conv_q1 = nn.Conv1d(proj_size, proj_size // 4, 1)
        self.conv_k1 = nn.Conv1d(proj_size, proj_size // 4, 1)
        # 以上是两个一维卷积层，用于计算注意力权重   ———— 注意力在pytorch中是用卷积实现的哦
        self.alpha1 = nn.Parameter(torch.zeros(1))  # 可训练参数 表示注意力机制的权重

        # merge information
        self.param_pool1 = nn.Linear(num_joints, 1)

        # class predict
        self.dropout = nn.Dropout(p=0.1)  # 随机使一些神经元失活 防止过拟合
        self.classifier = nn.Linear(proj_size, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        """
        b, c, h, w = x.size()
        hs = x.reshape(b,c,-1) # torch.Size([1, 288, 256])
        # pdb.set_trace()

        hs = self.param_pool0(hs)
        # print('after pool', hs.size())
        # torch.Size([2, 1536, 85])
        # adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)
        # print('q1', q1.size())
        # torch.Size([2, 85])
        k1 = self.conv_k1(hs).mean(1)
        # print('k1', k1.size())
        # torch.Size([2, 85])
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        # print('A1', A1.size())
        A1 = self.adj1 + A1 * self.alpha1
        # torch.Size([2, 85, 85])

        '''
        简单说就是 特征通过两个卷积层q1和k1 得到一个三维的张量，然后在第二个维度上取一个平均值
        然后通过pytorch的一个广播机制，将他们相减，得到一个差矩阵A1
        然后将这个矩阵和我们之前定义的一个学习的邻接矩阵相加， 这边A1还乘以了一个可学习权重α
        这个A1使用了tanh作为激活函数 就是限制注意力矩阵的每个元素在[-1,1]之间，函数图像
        '''
        # graph convolution
        hs = self.conv1(hs)
        # print('graph c', hs.size())
        # torch.Size([2, 1536, 85])
        hs = torch.matmul(hs, A1)
        # print('matmul', hs.size())
        # torch.Size([2, 1536, 85])
        hs = self.batch_norm1(hs)
        # print('BN', hs.size())
        # torch.Size([2, 1536, 85])
        '''
        特征(三维张量)经过一个卷积层后与注意力矩阵A1相乘
        这表示经过注意力机制加权后的一个特征向量，这么一个操作是增强模型对重要位置的关注
        抑制无关位置的影响，然后通过一个批归一化。
        '''
        # predict
        hs = self.param_pool1(hs)
        # print('pred', hs.size())
        # torch.Size([2, 1536, 1])
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        # print('fla', hs.size())
        # torch.Size([2, 1536])
        hs = self.classifier(hs)
        # print('class', hs.size())
        # torch.Size([2, 20])
        return hs

class Yin(nn.Module):
    def __init__(self, layers, embed_dims=None,
                mlp_ratios=None, downsamples=None,
                norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                num_classes=2000,
                in_patch_size=4, in_stride=4, in_pad=0,
                down_patch_size=2, down_stride=2, down_pad=0,
                drop_rate=0., drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5,
                fork_feat=True,
                init_cfg=None,
                pretrained=None,
                # the parameters for context-cluster
                proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32],  block=RepVGGBlock,
                **kwargs):
        super().__init__()

        self.backbone = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,fork_feat=True,
        **kwargs)
        fusion_in  = sum(embed_dims)
        embed_dim_p = 192
        fuse_block_num = 3
        proj_size = sum(embed_dims[0:2])
        select_num = 196 # 就是第三阶段输出特征图的大小 到时候改为可以自己选择
        # ================== Gold_Yolo ================
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(fusion_in, embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
                Conv(embed_dim_p, sum(embed_dims[0:2]), kernel_size=1, stride=1, padding=0),
        )
        # =================== 构造 GCN =====================
        self.combiner = GCNCombiner(num_classes, select_num, proj_size)


    def forward(self, x):
        # pdb.set_trace()
        x = self.backbone(x) # Context Cluster 输出的是 B C H W     而FPN 需要的输入是 B (H W) C 
        # print(x['layer1'].size()) torch.Size([32, 3136, 32])
        # print(x['layer2'].size()) torch.Size([32, 784, 64])
        # print(x['layer3'].size()) torch.Size([32, 784, 64])
        # print(x['layer4'].size()) torch.Size([32, 49, 320])
        x = self.low_FAM(x) # torch.Size([1, 1056, 16, 16])
        x = self.low_IFM(x) # torch.Size([1, 288, 16, 16])
        # ============= 加入GCNCombiner ===============
        comb_outs = self.combiner(x)
        # print(comb_outs.size())
        return comb_outs
        # logits['comb_outs'] .
        # = 


@register_model
def coc_tiny(pretrained=False, **kwargs):     
    layers = [3, 4, 5, 2]
    norm_layer = GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1

    model = Yin(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_tiny2(pretrained=False, **kwargs):
    layers = [3, 4, 5, 2]
    norm_layer = GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 2, 7, 4]
    proposal_h = [4, 2, 7, 4]
    fold_w = [7, 7, 1, 1]
    fold_h = [7, 7, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_small(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [4, 4, 8, 8]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_medium(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [6, 6, 12, 12]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_base_dim64(pretrained=False, **kwargs):
    layers = [6, 6, 24, 6]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [8, 8, 16, 16]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_base_dim96(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    norm_layer = GroupNorm
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [8, 8, 16, 16]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


"""
Updated: add plain models (without region partition) for tiny, small, and base , etc.
Re-trained with new implementation (PWconv->MLP for faster training and inference), achieve slightly better performance.
"""
@register_model
def coc_tiny_plain(pretrained=False, **kwargs):
    # sharing same parameters as coc_tiny, without region partition.
    layers = [3, 4, 5, 2]
    norm_layer = GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


if has_mmdet:
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_small_feat2(ContextCluster):
        def __init__(self, **kwargs):
                layers = [2, 2, 6, 2]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[2,2,2,2]
                proposal_h=[2,2,2,2]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[4,4,8,8]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_small_feat5(ContextCluster):
        def __init__(self, **kwargs):
                layers = [2, 2, 6, 2]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[5,5,5,5]
                proposal_h=[5,5,5,5]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[4,4,8,8]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_small_feat7(ContextCluster):
        def __init__(self, **kwargs):
                layers = [2, 2, 6, 2]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[7,7,7,7]
                proposal_h=[7,7,7,7]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[4,4,8,8]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat2(ContextCluster):
        def __init__(self, **kwargs):
                layers = [4, 4, 12, 4]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[2,2,2,2]
                proposal_h=[2,2,2,2]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[6,6,12,12]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat5(ContextCluster):
        def __init__(self, **kwargs):
                layers = [4, 4, 12, 4]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[5, 5, 5, 5]
                proposal_h=[5, 5, 5, 5]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[6,6,12,12]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat7(ContextCluster):
        def __init__(self, **kwargs):
                layers = [4, 4, 12, 4]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[7,7,7,7]
                proposal_h=[7,7,7,7]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[6,6,12,12]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = coc_base_dim64()
    out = model(input)
    print(model)
    print(out.shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters/1024**2))
