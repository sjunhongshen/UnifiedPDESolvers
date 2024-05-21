import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sklearn import metrics
import operator
from itertools import product
from functools import reduce, partial
from timm.models.layers import trunc_normal_
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denormalize(pred, target, mean, std):
    mean = mean.to(pred.device)
    std = std.to(pred.device)
    pred = (pred * std) + mean
    target = (target * std) + mean
    return pred, target


class MMD_loss(nn.Module):
    def __init__(self, src_data, maxsamples, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

        self.src_data = src_data
        self.src_data_len = len(src_data)

    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if self.fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def guassian_kernel_numpy(self, source, target):
        
        n_samples = int(source.shape[0])+int(target.shape[0])
        total = np.concatenate([source, target], 0)

        total0 = np.broadcast_to(np.expand_dims(total, 0), (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        total1 = np.broadcast_to(np.expand_dims(total, 1), (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        L2_distance = ((total0-total1)**2).sum(2) 
        if self.fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = np.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [np.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, target):
        if len(target.shape) > 2:
            target = target.mean(1)

        target_len = len(target)
        indices = np.random.choice(self.src_data_len, size=target_len)
        if torch.is_tensor(target):
            source = self.src_data[indices].to(target.device)
            kernels = self.guassian_kernel(source, target)
        else:
            source = self.src_data[indices]
            kernels = self.guassian_kernel_numpy(source, target)

        XX = kernels[:target_len, :target_len]
        YY = kernels[target_len:, target_len:]
        XY = kernels[:target_len, target_len:]
        YX = kernels[target_len:, :target_len] 

        loss = torch.mean(XX + YY - XY - YX) if torch.is_tensor(target) else np.mean(XX + YY - XY - YX)
        return loss


def nrmse(pred, target):
    idxs = target.size()
    nb = idxs[0]

    err_mean = torch.sqrt(torch.mean((pred.reshape([nb, -1]) - target.reshape([nb, -1])) ** 2, dim=1))
    err_MSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.reshape([nb, -1]) ** 2, dim=1))
    err_nrMSE = torch.mean(err_mean / nrm, dim=0)
    return err_nrMSE
    

def nrmse_loss(pred, target):
    idxs = target.size()
    nb = idxs[0]

    err_mean = torch.sqrt(torch.mean((pred.reshape([nb, -1]) - target.reshape([nb, -1])) ** 2, dim=1))
    err_MSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.reshape([nb, -1]) ** 2, dim=1))
    err_nrMSE = torch.mean(err_mean / nrm, dim=0)

    return err_nrMSE

def rmse_loss(pred, target):
    idxs = target.size()
    nb = idxs[0]

    # MSE
    err_mean = torch.sqrt(torch.mean((pred.reshape([nb, -1]) - target.reshape([nb, -1])) ** 2, dim=1))
    err_MSE = torch.mean(err_mean, axis=0)
    return err_MSE

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    res = res[0] if len(res) == 1 else res
    return res

class inverse_score(object):
    def __init__(self, score_func):
        self.score_func = score_func

    def __call__(self, output, target):
        return 1 - self.score_func(output, target)


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def set_grad_state(module, state):
    for n, m in module.named_modules():
        if len(n) == 0: continue
        if not state and 'position' in n: continue
        if not state and 'tunable' in n: continue
        for param in m.parameters():
            param.requires_grad = state


def set_param_grad(model, finetune_method):

    if finetune_method == "layernorm":
        for n, m in model.named_parameters():
            if 'layer' in n:
                if 'layernorm' in n or 'LayerNorm' in n:
                    continue
                else:
                    m.requires_grad = False

    elif finetune_method == "non-attn":
        for n, m in model.named_parameters():
            if 'layer' in n:
                if 'query' in n or 'key' in n or 'value' in n:
                    m.requires_grad = False


def get_params_to_update(model, finetune_method):

    params_to_update = []
    name_list = ''
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            name_list += "\t" + name

    print("Params to learn:", name_list)
    
    return params_to_update

def create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx=1):
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
    return position_ids.unsqueeze(0).expand(input_shape)


class adaptive_pooler(torch.nn.Module):
    def __init__(self, out_channel=1, output_shape=None, dense=False):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool1d(out_channel)
        self.out_channel = out_channel
        self.output_shape = output_shape
        self.dense = dense

    def forward(self, x):
        if len(x.shape) == 3:
            if self.out_channel == 1 and not self.dense:
                x = x.transpose(1, 2)
            pooled_output = self.pooler(x)
            if self.output_shape is not None:
                pooled_output = pooled_output.reshape(x.shape[0], *self.output_shape)
            else:
                pooled_output = pooled_output.reshape(x.shape[0], -1)
            
        else:
            b, c, h, w = x.shape
            x = x.reshape(b, c, -1)
            pooled_output = self.pooler(x.transpose(1, 2))
            pooled_output = pooled_output.transpose(1, 2).reshape(b, self.out_channel, h, w)
            if self.out_channel == 1:
                pooled_output = pooled_output.reshape(b, h, w)

        return pooled_output


class embedder_placeholder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is not None:
            return x

        return inputs_embeds


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def embedder_init(source, target, train_embedder=False, match_stats=False):
    if train_embedder:
        if hasattr(source, 'patch_embeddings'):
            if match_stats:
                weight_mean, weight_std = source.patch_embeddings.projection.weight.mean(), source.patch_embeddings.projection.weight.std()
                nn.init.normal_(target.projection.weight, weight_mean, weight_std)
                
                bias_mean, bias_std = source.patch_embeddings.projection.bias.mean(), source.patch_embeddings.projection.bias.std()
                nn.init.normal_(target.projection.bias, bias_mean, bias_std)
            else:
                rep_num = target.projection.in_channels // source.patch_embeddings.projection.in_channels + 1
                rep_weight = torch.cat([source.patch_embeddings.projection.weight.data] * rep_num, 1)
                
                target.projection.weight.data.copy_(rep_weight[:, :target.projection.in_channels, :target.projection.kernel_size[0], :target.projection.kernel_size[1]])        
                target.projection.bias.data.copy_(source.patch_embeddings.projection.bias.data)

            target.norm.weight.data.copy_(source.norm.weight.data)
            target.norm.bias.data.copy_(source.norm.bias.data)

        elif hasattr(source, 'patch_embedding'):
            if match_stats:
                weight_mean, weight_std = source.patch_embedding.weight.mean(), source.patch_embedding.weight.std()
                nn.init.normal_(target.projection.weight, weight_mean, weight_std)

                bias_mean, bias_std = source.patch_embedding.bias.mean(), source.patch_embedding.bias.std()
                nn.init.normal_(target.projection.bias, bias_mean, bias_std)
            else:
                rep_num = target.patch_embedding.in_channels // source.patch_embedding.in_channels + 1
                rep_weight = torch.cat([source.patch_embedding.weight.data] * rep_num, 1)

                target.patch_embedding.weight.data.copy_(rep_weight[:, :target.patch_embedding.in_channels, ::source.patch_embedding.kernel_size[0]//target.patch_embedding.kernel_size[0], ::source.patch_embedding.kernel_size[1]//target.patch_embedding.kernel_size[1]])
                target.patch_embedding.bias.data *= 0#.copy_(source.patch_embedding.bias.data)

                rep_num = target.position_embedding.weight.data.shape[0] // source.position_embedding.weight.data[1:,...].shape[0] + 1
                rep_weight = torch.cat([source.position_embedding.weight.data[1:,...]] * rep_num, 0)
                target.position_embedding.weight.data.copy_(rep_weight[:target.position_embedding.weight.data.shape[0],:])
            target.class_embedding.data.copy_(source.class_embedding.data)

        else:
            target.norm.weight.data.copy_(source.LayerNorm.weight.data)
            target.norm.bias.data.copy_(source.LayerNorm.bias.data)
            target.position_embeddings = copy.deepcopy(source.position_embeddings)
            target.norm2.weight.data.copy_(source.LayerNorm.weight.data)
            target.norm2.bias.data.copy_(source.LayerNorm.bias.data)

    else:
        for n, m in target.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)  

        try:
            target.position_embeddings = copy.deepcopy(source.position_embeddings)
        except:
            pass

def count_params(model):
    c = 0
    for p in model.parameters():
        try:
            c += reduce(operator.mul, list(p.size()))
        except:
            pass

    return c

def count_trainable_params(model):
    c = 0
    for p in model.parameters():
        try:
            if p.requires_grad:
                c += reduce(operator.mul, list(p.size()))
        except:
            pass

    return c
