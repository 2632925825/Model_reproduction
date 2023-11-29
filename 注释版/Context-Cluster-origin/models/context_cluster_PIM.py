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

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F
from pim_module import FPN, GCNCombiner, WeaklySelector, PluginMoodel


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
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    # 计算两个张量间的余弦相似度 返回一个矩阵 每个元素对应的是其相似度值
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """
        这段代码实现了一个聚类模型，用于将输入的特征图进行聚类操作。
        聚类过程包括计算相似度矩阵、利用相似度矩阵进行聚类分配和聚类后的特征向量计算。
        通过调整模型的参数，可以控制聚类的方式和输出的特征图通道数。
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
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        # print('out', out.size())
        # out torch.Size([16, 32, 56, 56])
        # out torch.Size([16, 64, 28, 28])
        # out torch.Size([16, 196, 14, 14])
        # out torch.Size([16, 320, 7, 7])
        # 应该是四个层依次采样
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
        # print('MLP', x.size())
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
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
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


class FPN(nn.Module):

    def __init__(self, inputs: dict, fpn_size: int, proj_type: str, upsample_type: str):
        """
        inputs : dictionary contains torch.Tensor
                 which comes from backbone output
        fpn_size: integer, fpn
        proj_type:
            in ["Conv", "Linear"]
        upsample_type:
            in ["Bilinear", "Conv", "Fc"]
            for convolution neural network (e.g. ResNet, EfficientNet), recommand 'Bilinear'.
            for Vit, "Fc". and Swin-T, "Conv"
        """
        super(FPN, self).__init__()

        self.fpn_size = fpn_size
        self.upsample_type = upsample_type
        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            # projection module
            if proj_type == "Conv":
                m = nn.Sequential(
                    nn.Conv2d(inputs[node_name].size(1), inputs[node_name].size(1), 1),
                    nn.ReLU(),
                    nn.Conv2d(inputs[node_name].size(1), fpn_size, 1)
                )
            elif proj_type == "Linear":
                m = nn.Sequential(
                    nn.Linear(inputs[node_name].size(-1), inputs[node_name].size(-1)),
                    nn.ReLU(),
                    nn.Linear(inputs[node_name].size(-1), fpn_size),
                )
            self.add_module("Proj_" + node_name, m)

            # upsample module
            if upsample_type == "Conv" and i != 0:
                assert len(inputs[node_name].size()) == 3  # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i - 1]].size(1)
                if in_dim != out_dim:
                    m = nn.Conv1d(in_dim, out_dim, 1)  # for spatial domain
                else:
                    m = nn.Identity()
                self.add_module("Up_" + node_name, m)

        if upsample_type == "Bilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def upsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x1_name: str):
        """
        return Upsample(x1) + x1
        """
        if self.upsample_type == "Bilinear":
            if x1.size(-1) != x0.size(-1):
                print(x1.size(-1))
                print(x0.size(-1))
                x1 = self.upsample(x1)
                print(x1.size(-1))
        else:
            x1 = getattr(self, "Up_" + x1_name)(x1)
        print(x1.size(-1))
        print(x0.size(-1))
        return x1 + x0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        hs = []
        for i, name in enumerate(x):
            x[name] = getattr(self, "Proj_" + name)(x[name])
            hs.append(name)

        for i in range(len(hs) - 1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i - 1]
            x[x0_name] = self.upsample_add(x[x0_name],
                                           x[x1_name],
                                           x1_name)
        return x


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
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32],
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
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

    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                nn.Conv1d(fpn_size, fpn_size, 1),
                nn.BatchNorm1d(fpn_size),
                nn.ReLU(),
                nn.Conv1d(fpn_size, num_classes, 1)
            )
            self.add_module("fpn_classifier_" + name, m)

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
                size_before_merge = x_out.size()[:-2]
                size_last = x_out.size()[-2:]
                merged_x_out = x_out.view(size_before_merge + (-1,))
                outs.append(merged_x_out)
        if self.fork_feat:
            fea_dic = {}
            for i in range(4):
                key = f'layer{i + 1}'
                value = outs[i]
                fea_dic[key] = value
            # output the features of four stages for dense prediction
            # [128,32,56,56] , [128, 64, 28, 28] [128, 196, 14, 14] [128, 320, 7, 7]
            return fea_dic

        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # output features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # size: [128,2000]
        # for image classification
        return cls_out


class WeaklySelector(nn.Module):

    def __init__(self, inputs: dict, num_classes: int, num_select: dict):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
                [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
                Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
                S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]

        """
        super(WeaklySelector, self).__init__()

        self.num_select = num_select

        # self.fpn_size = fpn_size
        # build classifier
        # if self.fpn_size is None:
        self.num_classes = num_classes
        for name in inputs:
            fs_size = inputs[name].size()
            if len(fs_size) == 3:
                in_size = fs_size[2]
            elif len(fs_size) == 4:
                in_size = fs_size[1]
            m = nn.Linear(in_size, num_classes)
            self.add_module("classifier_l_" + name, m)
        '''
        inputs是一个字典，包含若干个torch.Tensor对象，表示来自骨干网络的隐藏特征，每个键值对对应一个层的特征。
        num_classes是一个整数，表示分类任务的类别数。num_select也是一个字典，表示每个层应该选择多少个候选项。
        在初始化时，该模型会根据输入的特征构建分类器。如果特征是三维的，即形状为[B, S, C]，
        则分类器会在最后一个维度上加一个全连接层，将特征映射到num_classes个输出。如果特征是四维的，
        即形状为[B, C, H, W]，则分类器会先将特征展开成[B, C, H*W]的形状，
        再在最后一个维度上加一个全连接层。这样可以保证每个层都有一个对应的分类器。
        '''

        # def select(self, logits, l_name):
        #     """
        #     logits: [B, S, num_classes]
        #     """
        #     probs = torch.softmax(logits, dim=-1)
        #     scores, _ = torch.max(probs, dim=-1)
        #     _, ids = torch.sort(scores, -1, descending=True)
        #     sn = self.num_select[l_name]
        #     s_ids = ids[:, :sn]
        #     not_s_ids = ids[:, sn:]
        #     return s_ids.unsqueeze(-1), not_s_ids.unsqueeze(-1)
        '''
        logits是一个三维张量，形状为[B, S, num_classes]，表示一个批次中B个序列中每个位置预测的类别分数，S是序列长度，num_classes是类别数。
        l_name是一个字符串，用于指定当前处理的是哪个层。
        该方法首先对logits进行softmax，得到每个类别的概率分布，然后在最后一个维度上取最大值，得到每个位置预测的最大概率及其对应的类别。
        接着，将这些最大概率按降序排列，并取前sn个作为选中的候选项，其中sn是一个与l_name相关的参数。
        最后，将选中的候选项和未被选中的候选项的索引分别返回，形状均为[B, sn, 1]
        '''

    def forward(self, x, logits=None):
        """
        x :
            dictionary contain the features maps which
            come from your choosen layers.
            size must be [B, HxW, C] ([B, S, C]) or [B, C, H, W].
            [B,C,H,W] will be transpose to [B, HxW, C] automatically.
        """
        '''
        在前向传播过程中，模型接受一个字典x，表示来自骨干网络的特征。对于每个层的特征，模型首先将其展开成[B, S, C]的形状，
        然后在分类器上进行前向传播，得到该层的类别分数logits[name]。接着，模型对每个样本计算分类器输出的概率分布，
        并根据num_select[name]选出概率最大的num_select[name]个类别作为候选项，同时将剩余的类别作为未选中项。
        这里的选择过程是通过对概率分布的排序实现的。
        最后，模型将选中项和未选中项的隐藏特征分别保存到selections[name]和drop_[name]中，
        并将选中项的类别分数和未选中项的类别分数分别保存到logits["select_" + name]和logits["drop_" + name]中。
        最终，模型返回一个字典selections，包含每个层的选中项的隐藏特征。
        这个就和创新点对应上了
        '''

        # if self.fpn_size is None:
        logits = {}
        selections = {}
        for name in x:
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                x[name] = x[name].view(B, C, H * W).permute(0, 2, 1).contiguous()
            C = x[name].size(-1)
            # if self.fpn_size is None:
            logits[name] = getattr(self, "classifier_l_" + name)(x[name])
            probs = torch.softmax(logits[name], dim=-1)
            selections[name] = []
            preds_1 = []
            preds_0 = []
            num_select = self.num_select[name]
            for bi in range(logits[name].size(0)):
                max_ids, _ = torch.max(probs[bi], dim=-1)
                confs, ranks = torch.sort(max_ids, descending=True)
                sf = x[name][bi][ranks[:num_select]]
                nf = x[name][bi][ranks[num_select:]]  # calculate
                selections[name].append(sf)  # [num_selected, C]
                preds_1.append(logits[name][bi][ranks[:num_select]])
                preds_0.append(logits[name][bi][ranks[num_select:]])

            selections[name] = torch.stack(selections[name])
            preds_1 = torch.stack(preds_1)
            preds_0 = torch.stack(preds_0)

            logits["select_" + name] = preds_1
            logits["drop_" + name] = preds_0
        return selections


from typing import Union


class GCNCombiner(nn.Module):

    def __init__(self,
                 total_num_selects: int,
                 num_classes: int,
                 inputs: Union[dict, None] = None,
                 proj_size: Union[int, None] = None, ):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()
        for name in inputs:
            if len(inputs[name].size()) == 4:
                in_size = inputs[name].size(1)
            elif len(inputs[name].size()) == 3:
                in_size = inputs[name].size(2)
            else:
                raise ValueError("The size of output dimension of previous must be 3 or 4.")
            m = nn.Sequential(
                nn.Linear(in_size, proj_size),
                nn.ReLU(),
                nn.Linear(proj_size, proj_size)
            )
            self.add_module("proj_" + name, m)
        self.proj_size = proj_size

        # build one layer structure (with adaptive module) 构建邻接矩阵所需的维度
        num_joints = total_num_selects // 16  # 特征长度除以32取余后 这个整数作为邻接矩阵的大小  这个超参数是可以调的
        '''
        如果选择特征的总长度为 total_num_selects，那么将其划分为 num_joints 个长度为 32 的子区间，
        比如 选择的特征个数是 2720 那么他将分成85个部分，每个部分由原来的32个元素构成 
        关于这个32 是做了消融实验后得到的一个超参数 表示输入进单层的GCN 
        因为第一层选择的原始特征多 
        所以对应图中其实是上到下依次减少呗 这些原始特征依次输入GCN中 GCN学习这个超节点的表示 也就是下面的85维的矩阵
        '''
        self.param_pool0 = nn.Linear(total_num_selects, num_joints)
        # 将输入特征的总长度映射到一个的固定长度的向量中
        A = torch.eye(num_joints) / 100 + 1 / 100  # 创建一个对角矩阵
        self.adj1 = nn.Parameter(copy.deepcopy(A))  # 可训练参数，表示图卷积层网络的邻接矩阵
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        # 表示输入特征的通道数，也就是卷积核的输入通道数，而输出通道数也为self.proj_size，卷积核大小为1，
        # 即只在序列的一个位置上进行卷积操作。这个卷积核的作用是对输入特征进行通道间的信息交换和整合，以增强模型的表达能力。
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)  # 对输入数据进行归一化 使得每个特征的均值为0 方差为1 第一个参数为输入张量的特征数

        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size // 4, 1)
        # 以上是两个一维卷积层，用于计算注意力权重   ———— 注意力在pytorch中是用卷积实现的哦
        self.alpha1 = nn.Parameter(torch.zeros(1))  # 可训练参数 表示注意力机制的权重

        # merge information
        self.param_pool1 = nn.Linear(num_joints, 1)

        # class predict
        self.dropout = nn.Dropout(p=0.1)  # 随机使一些神经元失活 防止过拟合
        self.classifier = nn.Linear(self.proj_size, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        """
        hs = []
        for name in x:
            hs.append(getattr(self, "proj_" + name)(x[name]))

        # 以上代码 对于每个特征，如果没有使用FPN 则通过之前定义的FFN将其映射到低维向量中；如果使用了FPN，则直接使用该特征
        hs = torch.cat(hs, dim=1).transpose(1,
                                            2).contiguous()  # B, S', C --> B, C, S  将特征在第二个维度拼接起来（就是一个个小块哈）然后将S和C互换一下位置
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
        特征(三维张量）经过一个卷积层后与注意力矩阵A1相乘
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
        '''
        通过一个参数池化层后，对这个向量进行一个一定概率失活的dropout，这样理论上是避免过拟合
        然后再第一个维度上展平，，再通过一个全连接层输出类别分数
        '''
        return hs


class Yin(nn.Module):

    def __init__(self, layers, embed_dims,norm_layer, mlp_ratios, downsamples, down_patch_size,
                 down_pad, proposal_w, proposal_h, fold_w, fold_h,heads,head_dim,
                 fork_feat,
                 num_classes=2000, use_fpn=True,
                 fpn_size=3136, proj_type='Linear',
                 upsample_type='Bilinear', use_combiner=True, use_selection=True,
                 num_selects={'layer1': 16, 'layer2': 32,
                              'layer3': 64, 'layer4': 128}, comb_proj_size=512,
                 **kwargs):
        super(Yin, self).__init__()
        self.mlp_ratios = mlp_ratios
        rand_in = torch.randn(1, 3, 224, 224)
        self.backbone = ContextCluster(layers, embed_dims, norm_layer,self.mlp_ratios, downsamples, down_patch_size, down_pad, proposal_w, proposal_h, fold_w, fold_h,heads, head_dim)
        outs = self.backbone(rand_in)

        # = = = = = FPN = = = = =
        # self.use_fpn = use_fpn
        # if self.use_fpn:
        #     self.fpn = AttentionFusion()
        #     self.build_fpn_classifier(outs, fpn_size, num_classes)
        #
        # self.fpn_size = fpn_size

        # = = = = = Selector = = = = =
        self.use_selection = use_selection
        if self.use_selection:
            # w_fpn_size = self.fpn_size if self.use_fpn else None  # if not using fpn, build classifier in weakly selector
            self.selector = WeaklySelector(outs, num_classes, num_selects)

        # = = = = = Combiner = = = = =
        self.use_combiner = use_combiner
        if self.use_combiner:
            assert self.use_selection, "Please use selection module before combiner"
            gcn_inputs, gcn_proj_size = outs, comb_proj_size  # redundant, fix in future
            total_num_selects = sum([num_selects[name] for name in num_selects])  # sum
            self.combiner = GCNCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size)

    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                nn.Conv1d(fpn_size, fpn_size, 1),
                nn.BatchNorm1d(fpn_size),
                nn.ReLU(),
                nn.Conv1d(fpn_size, num_classes, 1)
            )
            self.add_module("fpn_classifier_" + name, m)

    def forward_backbone(self, x):
        return self.backbone(x)

    def forward(self, x):

        logits = {}
        # 初始
        # 经过了聚类骨干了，输出了四个层的张量
        x = self.forward_backbone(x)

        # if self.use_fpn:
        #     x = self.fpn(x)
        #     print('use FPN !')
        #     self.fpn_predict(x, logits)

        if self.use_selection:
            selects = self.selector(x, logits)
            # print('use WK')

        if self.use_combiner:
            comb_outs = self.combiner(selects)
            logits['comb_outs'] = comb_outs
            # print('use Comb')
            # print('com', logits['comb_outs'].size())
            return logits['comb_outs']

        # if self.use_selection or self.fpn:
        #     return logits

        # original backbone (only predict final selected layer)
        for name in x:
            hs = x[name]

        if len(hs.size()) == 4:
            hs = F.adaptive_avg_pool2d(hs, (1, 1))
            hs = hs.flatten(1)
        else:
            hs = hs.mean(1)
        out = self.classifier(hs)
        logits['ori_out'] = logits
        # print('last', logits)
        return logits


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


# @register_model
# def coc_tiny(pretrained=False, **kwargs):
#     layers = [3, 4, 5, 2]
#     norm_layer = GroupNorm
#     embed_dims = [32, 64, 196, 320]
#     mlp_ratios = [8, 8, 4, 4]
#     downsamples = [True, True, True, True]
#     proposal_w = [2, 2, 2, 2]
#     proposal_h = [2, 2, 2, 2]
#     fold_w = [8, 4, 2, 1]
#     fold_h = [8, 4, 2, 1]
#     heads = [4, 4, 8, 8]
#     head_dim = [24, 24, 24, 24]
#     down_patch_size = 3
#     down_pad = 1
#     model = ContextCluster(
#         layers, embed_dims=embed_dims, norm_layer=norm_layer,
#         mlp_ratios=mlp_ratios, downsamples=downsamples,
#         down_patch_size=down_patch_size, down_pad=down_pad,
#         proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
#         heads=heads, head_dim=head_dim,
#         **kwargs)
#     model.default_cfg = default_cfgs['model_small']
#     return model


# @register_model
# def coc_tiny2(pretrained=False, **kwargs):
#     layers = [3, 4, 5, 2]
#     norm_layer = GroupNorm
#     embed_dims = [32, 64, 196, 320]
#     mlp_ratios = [8, 8, 4, 4]
#     downsamples = [True, True, True, True]
#     proposal_w = [4, 2, 7, 4]
#     proposal_h = [4, 2, 7, 4]
#     fold_w = [7, 7, 1, 1]
#     fold_h = [7, 7, 1, 1]
#     heads = [4, 4, 8, 8]
#     head_dim = [24, 24, 24, 24]
#     down_patch_size = 3
#     down_pad = 1
#     model = ContextCluster(
#         layers, embed_dims=embed_dims, norm_layer=norm_layer,
#         mlp_ratios=mlp_ratios, downsamples=downsamples,
#         down_patch_size=down_patch_size, down_pad=down_pad,
#         proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
#         heads=heads, head_dim=head_dim,
#         **kwargs)
#     model.default_cfg = default_cfgs['model_small']
#     return model

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
            super().__init__(
                layers, embed_dims=embed_dims, norm_layer=norm_layer,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                down_patch_size=down_patch_size, down_pad=down_pad,
                proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                heads=heads, head_dim=head_dim,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_small_feat5(ContextCluster):
        def __init__(self, **kwargs):
            layers = [2, 2, 6, 2]
            norm_layer = GroupNorm
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [8, 8, 4, 4]
            downsamples = [True, True, True, True]
            proposal_w = [5, 5, 5, 5]
            proposal_h = [5, 5, 5, 5]
            fold_w = [8, 4, 2, 1]
            fold_h = [8, 4, 2, 1]
            heads = [4, 4, 8, 8]
            head_dim = [32, 32, 32, 32]
            down_patch_size = 3
            down_pad = 1
            super().__init__(
                layers, embed_dims=embed_dims, norm_layer=norm_layer,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                down_patch_size=down_patch_size, down_pad=down_pad,
                proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                heads=heads, head_dim=head_dim,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_small_feat7(ContextCluster):
        def __init__(self, **kwargs):
            layers = [2, 2, 6, 2]
            norm_layer = GroupNorm
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [8, 8, 4, 4]
            downsamples = [True, True, True, True]
            proposal_w = [7, 7, 7, 7]
            proposal_h = [7, 7, 7, 7]
            fold_w = [8, 4, 2, 1]
            fold_h = [8, 4, 2, 1]
            heads = [4, 4, 8, 8]
            head_dim = [32, 32, 32, 32]
            down_patch_size = 3
            down_pad = 1
            super().__init__(
                layers, embed_dims=embed_dims, norm_layer=norm_layer,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                down_patch_size=down_patch_size, down_pad=down_pad,
                proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                heads=heads, head_dim=head_dim,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat2(ContextCluster):
        def __init__(self, **kwargs):
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
            super().__init__(
                layers, embed_dims=embed_dims, norm_layer=norm_layer,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                down_patch_size=down_patch_size, down_pad=down_pad,
                proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                heads=heads, head_dim=head_dim,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat5(ContextCluster):
        def __init__(self, **kwargs):
            layers = [4, 4, 12, 4]
            norm_layer = GroupNorm
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [8, 8, 4, 4]
            downsamples = [True, True, True, True]
            proposal_w = [5, 5, 5, 5]
            proposal_h = [5, 5, 5, 5]
            fold_w = [8, 4, 2, 1]
            fold_h = [8, 4, 2, 1]
            heads = [6, 6, 12, 12]
            head_dim = [32, 32, 32, 32]
            down_patch_size = 3
            down_pad = 1
            super().__init__(
                layers, embed_dims=embed_dims, norm_layer=norm_layer,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                down_patch_size=down_patch_size, down_pad=down_pad,
                proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                heads=heads, head_dim=head_dim,
                fork_feat=True,
                **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat7(ContextCluster):
        def __init__(self, **kwargs):
            layers = [4, 4, 12, 4]
            norm_layer = GroupNorm
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [8, 8, 4, 4]
            downsamples = [True, True, True, True]
            proposal_w = [7, 7, 7, 7]
            proposal_h = [7, 7, 7, 7]
            fold_w = [8, 4, 2, 1]
            fold_h = [8, 4, 2, 1]
            heads = [6, 6, 12, 12]
            head_dim = [32, 32, 32, 32]
            down_patch_size = 3
            down_pad = 1
            super().__init__(
                layers, embed_dims=embed_dims, norm_layer=norm_layer,
                mlp_ratios=mlp_ratios, downsamples=downsamples,
                down_patch_size=down_patch_size, down_pad=down_pad,
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
    print("number of params: {:.2f}M".format(n_parameters / 1024 ** 2))
