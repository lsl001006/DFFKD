import torch
import torch.nn as nn

def backward_hook(m, input_gradients, output_gradients):
    print('input_gradients {}'.format(input_gradients))
    print('output_gradients {}'.format(output_gradients))
    input_gradients = (torch.zeros_like(input_gradients[0]), )
    return input_gradients

conv = nn.Conv2d(1, 1, 3)
conv.register_full_backward_hook(backward_hook)

x = torch.randn(1, 1, 3, 3).requires_grad_()
print(x)
out = conv(x)
out.backward()
print(x.grad) # ones


