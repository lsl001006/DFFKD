import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_BOUND = 1.
SENSITIVITY = 2.

def master_hook_adder(module, grad_input, grad_output):
    '''
    global hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global dynamic_hook_function
    
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):
    '''
    dummy hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    print("dummy!")
    pass


def modify_gradnorm_conv_hook(module, grad_input, grad_output):
    '''
    gradient modification hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    return tuple(grad_input_new)


def dp_conv_hook(module, grad_input, grad_output):
    '''
    gradient modification + noise hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global noise_multiplier
    ### get grad wrt. input (image)
    # grad_input = [shape[8,64,14,14],None]
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize # 1/8=0.125

    grad_wrt_image = grad_wrt_image.view(batchsize, -1) # shape [8,64,14,14] -> [8,12544]
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1) # shape [8]

    ### clip
    
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image

    ### add noise
    # 正态分布的噪声
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
    grad_wrt_image = grad_wrt_image + noise
    grad_input_new = [grad_wrt_image.view(grad_input_shape)] # 转换成输入形式
    for i in range(len(grad_input) - 1):
        grad_input_new.append(grad_input[i + 1])
    # 返回clip + noise后的grad_input
    return tuple(grad_input_new)



def register_hooks(modules):
    hooks = []
    for m in modules:
        hooks.append( FeatureHook(m) )
    return hooks


class FeatureHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.output = output
        self.input = input[0]
    
    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


class FeatureMeanHook(object):
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        self.instance_mean = torch.mean(input[0], dim=[2, 3])

    def remove(self):
        self.hook.remove()

    def __repr__(self):
        return "<Feature Hook>: %s"%(self.module)


class FeatureMeanVarHook():
    def __init__(self, module, on_input=True, dim=[0,2,3]):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.on_input = on_input
        self.module = module
        self.dim = dim

    def hook_fn(self, module, input, output):
        # To avoid inplace modification
        if self.on_input:
            feature = input[0].clone() 
        else:
            feature = output.clone()
        self.var, self.mean = torch.var_mean( feature, dim=self.dim, unbiased=True )

    def remove(self):
        self.hook.remove()
        self.output=None


class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def remove(self):
        self.hook.remove()