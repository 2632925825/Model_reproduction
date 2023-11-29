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
import scipy.sparse as sp
import numpy as np 

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

def create_adj( H, W, C, neibour):
    """
    功能：
        根据featuremap的高和宽建立对应的空域邻接矩阵,
    输入：
        h featuremap的高度
        w featuremap的宽
        C featuremap的通道数 
        neibour  4或8决定空域adj的邻居数   2 决定计算channel的adj
    """
    h = H
    w = W
    n = h*w
    x = [] #保存节点
    y = [] #保存对应的邻居节点
    #判断是生成8邻居还是4邻居
    if neibour==8:
        l =np.reshape(np.arange(n),(h,w))
        # print(l)
        # print(((l[:,2])+w)[:1])
        #print(l[:,2])
        for i in range(h): 
            #邻界条件需要考虑，故掐头去尾先做中间再两边
            r = l[i,:]
            #左邻
            x = np.append(x,r[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(r-1)[1:]).astype(int)
            #每个元素的右邻居
            x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
            y = np.append(y,(r+1)[:-1]).astype(int) 
            if i >0:
                #上邻
                x = np.append(x,r).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w)).astype(int) 
                #左上
                x = np.append(x,r[1:]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w-1)[1:]).astype(int) 
                #右上
                x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w+1)[:-1]).astype(int) 
            if i <h-1:
                #下邻
                x = np.append(x,r).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w)).astype(int) 
                #左下
                x = np.append(x,r[1:]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w-1)[1:]).astype(int) 
                #右上
                x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w+1)[:-1]).astype(int)                           
    elif neibour==4:       #4邻居
        l =np.reshape(np.arange(n),(h,w))
        for i in range(h): 
            v = l[i,:]
            x = np.append(x,v[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(v-1)[1:]).astype(int)
            #每个元素的右邻居
            x = np.append(x,v[:-1]).astype(int) #最后一个没有下一个邻居
            y = np.append(y,(v+1)[:-1]).astype(int) 

        for i in range(w):
            p = l[:,i]
            #上邻
            x = np.append(x,p[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(p-w)[1:]).astype(int)
            #下邻  
            x = np.append(x,p[:-1]).astype(int) #0没有上一个邻居
            y = np.append(y,(p+w)[:-1]).astype(int)
    elif neibour==2:       #4邻居
        n = C
        l =np.arange(n)

        #每个元素的上一个邻居
        x = np.append(x,l[1:]).astype(int) #0没有上一个邻居
        y = np.append(y,(l-1)[1:]).astype(int)
        #每个元素的下一个邻居
        x = np.append(x,l[:-1]).astype(int) #最后一个没有下一个邻居
        y = np.append(y,(l+1)[:-1]).astype(int) 
    adj = np.array((x,y)).T  #生成的两列合并得到节点及其邻居的矩阵
    #print(adj)
    #使用sp.coo_matrix() 和 np.ones() 共同生成临界矩阵，右边的

    adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),shape=(n, n),dtype=np.float32)

    # build symmetric adjacency matrix 堆成矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = np.hstack((x,y)).rehsape(-1,2) 这样reshape得到的是临近两个一组变化成两列，不符合条件
    #adj =normalize( adj + sp.eye(adj.shape[0]))
    adj = np.array(adj.todense())
    ''''      保存adj的数据查看是什么形状      	'''
    # np.save('./adj.txt',x) 
    adj = torch.tensor(adj).cuda()
    # adj = torch.tensor(adj)

    #adj = torch.FloatTensor(x).cuda()
    return adj


class GraphAttentionLayer(nn.Module):
	"""
	描述：
		再MPGA中，单层的GAT，输入输出的维度相同，attention的计算方式使用softmax
		in_features：输入的维度，
		down_ratio:降维的比例
		out_feature:在多头注意力之中需要用
	"""
	def __init__(self, in_features,down_ratio=8,sgat_on=True,cgat_on=True):
		super(GraphAttentionLayer, self).__init__()
		#self.dropout = dropout
		self.in_features = in_features
		self.hid_features = in_features//down_ratio #数据降维，使用//保证输出的结果为整数
		#alpha sigma两次降维后，做矩阵运算获得att，类似GAT中的先用w再用a获得注意力。
		self.use_sgat = sgat_on
		self.use_cgat = cgat_on
		if self.use_sgat:
			self.down_alpha = nn.Sequential( #无论是通道还是空域都需要先降维 图卷积的#默认使用hid_feature是
				nn.Conv2d(in_channels=self.in_features, out_channels=self.hid_features, 
						kernel_size=1, stride=1, padding=0, bias=False), #输入 in_features 2048 输出hid_features  in_features//down_ratio
				nn.BatchNorm2d(self.hid_features),
				nn.ReLU()
			)

			self.down_sigma = nn.Sequential( #无论是通道还是空域都需要先降维 图卷积的
				nn.Conv2d(in_channels=self.in_features, out_channels=self.hid_features, 
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.hid_features),
				nn.ReLU()
			)
	

	def forward(self, x):#v 是输入的各个节点， b c h w是输入feature map的shape
		#输入后都是先降维计算注意力，concat=false需要聚合特征前需要将输入的维度降低再聚合
		b,c,h,w = x.size()

		if self.use_sgat:
			adj = create_adj(h,w,self.in_features,8)
			#print('图片的维度：',x.size())
			alpha = self.down_alpha(x)#concat的时候不太一样
			#print('alpha :',alpha.shape)
			#print('alpha.shape:',alpha.shape)
			sigma = self.down_sigma(x)
			#print('sigma :',sigma.shape)
			alpha = alpha.view(b, self.hid_features, -1).permute(0, 2, 1) #8 32 64*32	
			#print('转换后alpha :',alpha.shape)
			sigma = sigma.view(b, self.hid_features, -1)
			#print('转换后sigma :',sigma.shape)
			att = torch.matmul(alpha, sigma) #这就是每个图的自注意力机制
			#print('alpha乘sigma得到大的att shape:',att.shape)
			zero_vec = -9e15*torch.ones_like(att)
			attention = torch.where(adj.expand_as(att)> 0, att,zero_vec)
			#print('attention shape:',attention.shape)
			attention = F.softmax(attention, dim=2)  # 计算节点，临近节点的关系，因为attention为3维 b h w 按行求则为
			#print('softmax(attention) shape:',attention.shape)
			h_s = torch.matmul(attention, x.view(b, c, -1).permute(0, 2, 1)).permute(0,2,1).view(b,c,h,w)  #聚合临近节点的信息表示该节点
			#print('图上传播后',h_prime.shape)
		if self.use_cgat:
			cadj = create_adj(h,w,c,2)#2表示通道的adj未进行节点维度的变化，直接点乘和sigmod计算的att
			theta_xc = x.view(b, c, -1)
			phi_xc = x.view(b, c, -1).permute(0, 2, 1) # 8 2048 256 1  batch_size 节点 channel
			Gc = torch.matmul(theta_xc, phi_xc) # bactchsiz n n   通道之间的关系 
			zero_vec = -9e15*torch.ones_like(Gc)
			catt = torch.where(cadj.expand_as(Gc)> 0, Gc, zero_vec)
			cattention = F.softmax(catt, dim=2)  # 计算节点，临近节点的关系，因为attention为3维 b h w 按行求则为
			h_prime = torch.matmul(cattention, x.view(b, c, -1)).view(b,c,h,w)   #聚合临近节点的信息表示该节点
		if self.use_cgat and self.use_sgat:
			return torch.add(h_s, h_prime)
		if self.use_cgat:
			return h_prime #残差

		return  h_s


class GAT(nn.Module):
    def __init__(self, nfeat,down_ratio=8,sgat_on=True,cgat_on=True):
        """Dense version of GAT.
        描述：
            nfeat :输入的维度
            nclass ：非concat时使用的输出维度
            Height Width：输入的图片的大小
            dow_ratio:concat时输入数据的维度降低倍数
        """
        super(GAT, self).__init__()
        # self.nheads = 3
        # print ('Use_SGAT_Att: {};\tUse_CGAT_Att: {}.'.format(sgat_on, cgat_on))
        # self.attentions= nn.ModuleList([GraphAttentionLayer(nfeat,down_ratio=down_ratio,sgat_on=sgat_on,cgat_on=cgat_on) for _ in range(self.nheads)])
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.attentions= GraphAttentionLayer(nfeat,down_ratio=down_ratio,sgat_on=sgat_on,cgat_on=cgat_on)

        
        self.gama = nn.Parameter(torch.zeros(1))

        self.output_down = nn.Sequential( 
        nn.Conv2d(in_channels=nfeat, out_channels=nfeat, 
                kernel_size=1, stride=1, padding=0, bias=False), #输入 in_features 2048 输出hid_features  in_features//down_ratio
        nn.BatchNorm2d(nfeat),
		nn.ReLU()

		)

    def forward(self, x):#resnet输入的维度为 b 2048 16 8 
        b,c,h,w = x.size()
        # x = F.dropout(x, self.dropout, training=self.training)
        #     h_prime = torch.cat([att(x) for att in self.attentions], dim=1)  #通道直接拼接上还是 如果维度变为1024 则最终 b n 4096 考虑
        #     h_prime =  self.output_down(h_prime)#到这一步已经算是进行了两次的GCN  在这一部分中再次将维度恢复到2048
        h_prime =self.attentions(x)
        # if  self.nheads > 1 :
        #     h_prime = torch.cat([att(x) for att in self.attentions], dim=1)  #通道直接拼接上还是 如果维度变为1024 则最终 b n 4096 考虑
        #     h_prime =  self.output_down(h_prime)#到这一步已经算是进行了两次的GCN  在这一部分中再次将维度恢复到2048
        # else:
        #     h_prime =self.attention_0(x)
        #x = F.dropout(x, self.dropout, training=self.training) 
        #x = F.dropout(x, self.dropout, training=self.training) 
        h_prime = F.elu(h_prime) #因为输入x是经果relu的所有这里也需要经过这个然后输入res中
        h_prime = (1-self.gama)*x+self.gama*h_prime #残差
        # print('cancha ')
        return h_prime 

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
        self.return_center_dim = head_dim * heads
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.pga = GAT(self.return_center_dim)

    def forward(self, x):  # [b,c,w,h]
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
        # pdb.set_trace()
        centers = self.centers_proposal(x)  # torch.Size([B, 24, 2, 2]), we set M = C_W*C_H and N = w*h
        return_centers = rearrange(centers, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        return_centers = rearrange(return_centers, "(b e) c w h -> b (e c) w h", e=self.heads)
        return_centers = self.pga(return_centers)
        # return_centers = self.pga(return_centers)
        # return_centers = self.pga(return_centers)
        centers = rearrange(return_centers, "b (e c) w h -> (b e) c w h",e=self.heads)
        centers = rearrange(centers, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
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
                 fork_feat=False,
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
        heads=heads, head_dim=head_dim,fork_feat=False,
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
