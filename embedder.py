import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch
from torch import einsum
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from functools import partial
from transformers import AutoModel, AutoConfig, RobertaForTokenClassification, AutoTokenizer, DataCollatorWithPadding

from task_configs import get_data, get_optimizer_scheduler, get_optimizer
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, MMD_loss, get_params_to_update
import copy


def get_tgt_model(args, root, sample_shape, loss, maxsize=5000, alignment=True, eval_mode=False):
        
    if alignment:
        trainset = load_dataset("conll2003", split='validation')
        trainset = trainset.select_columns(['tokens'])
        tokenizer = AutoTokenizer.from_pretrained("roberta-base" if not args.size == "large" else "roberta-large")
            
        def preprocess_function(examples):
            examples["strs"] = ["".join(toks) for toks in examples["tokens"]]
            examples["input_ids"] = tokenizer(examples["strs"])['input_ids']
            del examples['tokens']
            del examples['strs']
            return examples
        
        trainset = trainset.map(preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer)
        del tokenizer 

        src_train_loader = DataLoader(trainset, batch_size=32, collate_fn=data_collator)
        
        src_model = ModelWrapper(sample_shape, use_embedder=False, weight=args.weight, size=args.size, train_epoch=args.embedder_epochs, drop_out=args.drop_out)
        src_model = src_model.to(args.device).eval()
        src_model.output_raw = "src"
                                                            
        src_feats = []
        src_ys = []
        for i, data in enumerate(src_train_loader):
            x_ = data['input_ids']
            x_ = x_.to(args.device)
            out = src_model(x_)

            if len(out.shape) > 2:
                out = out.mean(1)

            src_feats.append(out.detach().cpu())
            if len(src_feats) > maxsize:
                break

        src_feats = torch.cat(src_feats, 0)
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_feats)        
        del src_model, trainset, src_train_loader  
        torch.cuda.empty_cache()

    if not eval_mode:    
        tgt_train_loader, _, _, n_train, _, _ = get_data(root, args.dataset, args.batch_size, False, get_shape=True, args=args)

        print("src feat shape", src_feats.shape) 
   
    tgt_model = ModelWrapper(sample_shape, weight=args.weight, size=args.size, train_epoch=args.embedder_epochs, target_seq_len=args.target_seq_len, drop_out=args.drop_out, add_text=args.add_text)
    tgt_model = tgt_model.to(args.device).train()

    if eval_mode:
        tgt_model.output_raw = False
        for name, param in tgt_model.named_parameters():
            param.requires_grad = True
        return tgt_model, [] 

    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')

    tgt_model_optimizer.zero_grad()

    score_func = MMD_loss(src_data=src_feats, maxsamples=args.maxsamples)
    score = 0
    total_losses, times, embedder_stats = [], [], []
    
    for ep in range(args.embedder_epochs):   
        
        total_loss = 0 
        total_loss1 = 0
        total_loss2 = 0
        
        feats = []
        feats2 = []
        ys = []

        datanum = 0
        
        time_start = default_timer()
        for j, data in enumerate(tgt_train_loader):
                
            x, y = data

            if isinstance(x, list):
                x, text_embeddings = x
                text_embeddings = text_embeddings.to(args.device)
            else:
                text_embeddings = None
                
            if isinstance(y, list):
                y, mask = y
                y = y.to(args.device)
                mask = mask.to(args.device)
                y *= mask
            else:
                y = y.to(args.device)
                mask = None

            x = x.to(args.device)
            out, xfno = tgt_model(x, text_embeddings=text_embeddings)

            if mask is not None:
                xfno *= mask
            
            feats.append(xfno)
            feats2.append(out)                
            ys.append(y)
                
            datanum += x.shape[0]
                
            if datanum > args.maxsamples or j == len(tgt_train_loader) - 1: 
                    
                feats = torch.cat(feats, 0)
                feats2 = torch.cat(feats2, 0).mean(1)
                ys = torch.cat(ys, 0)
 
                loss1 = loss(feats, ys)
                loss2 = score_func(feats2)

                loss_all = loss1 + loss2

                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss += loss_all.item()
                
                loss_all.backward()
                tgt_model_optimizer.step()
                tgt_model_optimizer.zero_grad()
                    
                feats = []
                feats2 = []
                ys = []
                datanum = 0
                    
        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\tloss all:", "%.4f" % total_losses[-1], "\tl task:", "%.4f" % total_loss1, "\tl align:", "%.4f" % total_loss2)

        tgt_model_scheduler.step()

    del tgt_train_loader

    tgt_model.output_raw = False
    
    for name, param in tgt_model.named_parameters():
        param.requires_grad = True

    return tgt_model, embedder_stats


class ModelWrapper(torch.nn.Module):
    def __init__(self, input_shape, use_embedder=True, weight='roberta', size="base", train_epoch=0, target_seq_len=512, drop_out=None, add_text="cross", from_scratch=False):
        super().__init__()

        self.output_raw = True
        self.weight = weight
        
        if size == "large":
            modelname = "roberta-large"
            embed_dim = 1024
        else:
            modelname = 'roberta-base' 
            embed_dim = 768
        configuration = AutoConfig.from_pretrained(modelname)
        if drop_out is not None:
            configuration.hidden_dropout_prob = drop_out
            configuration.attention_probs_dropout_prob = drop_out
        self.model = AutoModel.from_pretrained(modelname, config=configuration) if not from_scratch else AutoModel.from_config(configuration)

        if use_embedder:
            self.embedder = PDEEmbeddings(input_shape, config=self.model.config, embed_dim=embed_dim, target_seq_len=target_seq_len, add_text=add_text)
            embedder_init(self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)    
        else:
            self.embedder = self.model.embeddings

        self.model.embeddings = embedder_placeholder()
        self.model.pooler = nn.Identity()
        self.predictor = nn.Linear(in_features=embed_dim, out_features=4 * 128 * 128)  

        conv_init(self.predictor) 
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, True)
    

    def forward(self, x, text_embeddings=None):

        if self.output_raw:
            if self.output_raw == "src":
                return self.embedder(x)
            return self.embedder(x, raw=True, text_embeddings=text_embeddings) 

        x, _ = self.embedder(x, text_embeddings=text_embeddings)

        x = self.model(inputs_embeds=x)['last_hidden_state']
        x = self.predictor(x.mean(1, keepdim=True)).reshape(-1,4,128,128)

        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=12, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class PDEEmbeddings(nn.Module):
    def __init__(self, input_shape, embed_dim=768, target_seq_len=64, config=None, add_text="cross"):
        super().__init__()
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)

        self.projection = nn.Conv1d(128 ** 2, embed_dim, kernel_size=1)
        conv_init(self.projection)
        self.norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fno = FNO2d()

        if add_text == "cross":
            self.cro_attn = CrossAttention(embed_dim, embed_dim)
        self.add_text = add_text


    def get_stack_num(self, input_len, target_seq_len):
        for i in range(1, input_len + 1):
            if input_len % i == 0 and input_len // i <= target_seq_len:
                break
        return i


    def forward(self, x=None, raw=False, text_embeddings=None):

        xfno, x = self.fno(x)
        x = self.projection(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.norm(x)

        if text_embeddings is not None:
            if self.add_text == "cross":
                x = self.cro_attn(x, text_embeddings)
            elif self.add_text == "cat":
                x = torch.cat([text_embeddings, x],1)
            else:
                x = x
            
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps

        x = self.norm2(x)
        
        return x, xfno


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, num_channels=4, modes1=12, modes2=12, width=32):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x, y, c)
        output: the solution of the next timestep
        output shape: (batchsize, x, y, c)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width 
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(num_channels + 2, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # Pad tensor with boundary condition
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2   

        x = x[..., :-self.padding, :-self.padding] # Unpad the tensor

        x_raw = x.reshape(x.shape[0], x.shape[1], -1)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x).permute(0,3,1,2)

        return x,x_raw