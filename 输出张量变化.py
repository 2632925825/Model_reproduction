import timm 
import torch
from torch import nn

def print_shape(m, i, o):
    # 版本1
    print(m, i[0].shape, o.shape)
    # 版本2
    print(i[0].shape, '=>', o.shape)


def get_children(model: nn.Module):
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


model = None
# 版本1
flatt_children = get_children(model)
for layer in flatt_children:
    layer.register_forward_hook(print_shape)

# 版本2  
for layer in model.children():
    layer.register_forward_hook(print_shape)

batch_input = torch.randn(4, 3, 224, 224)
model(batch_input)