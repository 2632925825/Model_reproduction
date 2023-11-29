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
        x_l, x_m, x_s, x_n = x #  这里需要修改一下 x_n是特征图最小的那个 需要用到插值方法 统一S
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([x_l, x_m, x_s, x_n], 1) # 在第一个channel数进行拼接 (B, C, S)
        return out

class GCNCombiner(nn.Module):

    def __init__(self):
        """
        If building backbone without FPN, set fpn_size to None and MUST give
        'inputs' and 'proj_size', the reason of these setting is to constrain the
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        total_num_selects = 256
        self.proj_size = 288
        num_classes = 2000
        # build one layer structure (with adaptive module) 构建邻接矩阵所需的维度
        num_joints = total_num_selects // 16  # 特征长度除以32取余后 这个整数作为邻接矩阵的大小  这个超参数是可以调的
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

class Gold_Com(nn.Module):
    def __init__(self, fusion_in, embed_dim_p, fuse_block_num, trans_channels, block=RepVGGBlock):
        super(Gold_Com, self).__init__()
        self.low_FAM = SimFusion_4in()
        self.low_IFM = nn.Sequential(
                Conv(fusion_in, embed_dim_p, kernel_size=1, stride=1, padding=0),
                *[block(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
                Conv(embed_dim_p, sum(trans_channels[0:2]), kernel_size=1, stride=1, padding=0),
        )
        self.com = GCNCombiner()

    def forward(self, x):
        # pdb.set_trace()
        x = self.low_FAM(x) # torch.Size([1, 1056, 16, 16])
        x = self.low_IFM(x) # torch.Size([1, 288, 16, 16])
        x = self.com(x)
        print(x.shape)
        return x




if __name__ == '__main__':
    # torch.Size([32, 32, 56, 56])
    # torch.Size([32, 64, 28, 28])
    # torch.Size([32, 196, 14, 14])  --> 196 转化为图矩阵
    # torch.Size([32, 320, 7, 7])
    fusion_in = 1056
    embed_dim_p = 192
    fuse_block_num=3
    trans_channels=[192, 96, 192, 384] 
    model = Gold_Com(fusion_in, embed_dim_p, fuse_block_num, trans_channels)
    inputs = [torch.randn(1, 32, 56, 56),torch.randn(1, 64, 32, 32),torch.randn(1, 320, 16, 16),torch.randn(1, 640, 8, 8)]
    out = model(inputs)