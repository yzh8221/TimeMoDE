import torch
import torch.nn as nn
import numpy as np
import math
from .timm import Attention
from .Cluster import EDESC
from utils import persistence

import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x
    
@persistence.persistent_class
class DynamicLinear(torch.nn.Module):
    def __init__(self, dynamic_size=[128,128], out_channels=None, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        
        self.dynamic_size = dynamic_size
        self.out_channels = out_channels
        init_kwargs = dict(mode=init_mode, fan_in=self.dynamic_size[0], fan_out=self.dynamic_size[1])
        self.dynamic_weights = torch.nn.Parameter(weight_init([1, 1, dynamic_size[0], dynamic_size[1]], **init_kwargs) * init_weight)
        self.dynamic_bias = torch.nn.Parameter(weight_init([self.dynamic_size[1]], **init_kwargs) * init_bias)

    def forward(self, x, out_features=None):
        if out_features is None and self.out_channels is None:
            raise Exception("out_features & self.out_channels cannot both be None")
        out_features = self.out_channels if self.out_channels is not None else out_features

        weight = torch.nn.functional.interpolate(self.dynamic_weights, size=(out_features, x.size(2)), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
        bias = torch.nn.functional.interpolate(self.dynamic_bias.unflatten(0, [1,1,1,-1]), size=(1, out_features), mode='bicubic', align_corners=False).flatten()
        
        x = torch.matmul(x, weight.T)
        x = x + bias
        return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # if inference with fp16, embedding.half()
        return embedding 

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size) 
        t_emb = self.mlp(t_freq)#.half())
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                MoE Layer.                                     #
#################################################################################


class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01, eta=5.0, cluster_gamma=0.01, finetune=True):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.gamma = cluster_gamma
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

        # Cluster Prototype
        self.cluster = EDESC(d_model=self.gating_dim, n_clusters=self.n_routed_experts, eta=eta)

        if finetune:
            self.freeze_cluster()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def freeze_cluster(self):
        for param in self.cluster.parameters():
            param.requires_grad = False
    
    def forward(self, hidden_states, xo):

        bsz, seq_len, h = hidden_states.shape 
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        xo = xo.unsqueeze(1).repeat(1, seq_len, 1)
        xo = xo.view(-1, h)
        logits_xo = self.cluster(xo)
        cluster_loss = self.cluster.total_loss(self.n_routed_experts)

        logits = logits + logits_xo

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        # select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss, cluster_loss*self.gamma


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.pretraining_tp = pretraining_tp

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3*hidden_size, bias=True)
        )

    def forward(self, x, c):

        identity = x
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift

        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0) 
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        down_proj = identity + gate * down_proj
        return down_proj


class SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, pretraining_tp=2, gamma=0.01, finetune=True):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.finetune = finetune
        self.experts = nn.ModuleList([MoeMLP(hidden_size = embed_dim, intermediate_size = mlp_ratio * embed_dim, pretraining_tp=pretraining_tp) for i in range(num_experts)])
        self.gate = MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, cluster_gamma=gamma, finetune=self.finetune)
        self.n_shared_experts = 1
        
        if self.n_shared_experts is not None:
            intermediate_size =  embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size = embed_dim, intermediate_size = intermediate_size, pretraining_tp=pretraining_tp)
    
    def forward(self, hidden_states, c, xo):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss, cluster_loss = self.gate(hidden_states, xo) 

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        c = c.repeat_interleave(int(orig_shape[1]), dim=0)
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            _c = c.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i], _c[flat_topk_idx == i]).float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            if not self.finetune:
                y = AddAuxiliaryLoss.apply(y, aux_loss)
                y = AddAuxiliaryLoss.apply(y, cluster_loss)
        else:
            y = self.moe_infer(hidden_states, c, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity, c.reshape(orig_shape[0], orig_shape[1], -1))
        return y
    

    @torch.no_grad()
    def moe_infer(self, x, c, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_c = c[exp_token_idx]
            expert_out = expert(expert_tokens, expert_c)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

#################################################################################
#                                 Embedding layer                               #
#################################################################################


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(1, max_len, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)
    

class TimeSeries2EmbLinear(nn.Module):
    """
    Encode time series data alone with selected dimension.
    """

    def __init__(
        self,
        hidden_size=512,
        shape=(24, 64),
        dropout=0,
    ):
        super().__init__()
        sequence_length, feature_size = shape
        self.pos_emb = LearnablePositionalEncoding(d_model=hidden_size, max_len=sequence_length)
        self.processing = nn.Sequential(nn.Linear(feature_size, hidden_size), nn.Dropout(dropout))

    def forward(self, x):
        x = self.processing(x)
        return self.pos_emb(x)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4, num_experts=8, 
        num_experts_per_tok=2, pretraining_tp=2, gamma=0.01, finetune=True, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.moe = SparseMoeBlock(hidden_size, mlp_ratio, num_experts, num_experts_per_tok, pretraining_tp, gamma, finetune)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, xo):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) 
        x = x + gate_mlp.unsqueeze(1) * self.moe(modulate(self.norm2(x), shift_mlp, scale_mlp), c, xo)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_len=24,
        in_channels=64,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        num_experts=8, num_experts_per_tok=2,
        pretraining_tp=2,
        dropout=0,
        dynamic_size=[32, 128],
        gamma=0.01,
        patch_len=1,
        finetune=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.patch_len = patch_len
        patch_num = int(in_len/patch_len)
        patch_dim = int(in_channels*patch_len)

        # DyConv
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        self.dynamic_size = dynamic_size
        self.emb_conv = DynamicLinear(dynamic_size=self.dynamic_size, out_channels=in_channels, **init)

        # embedder
        self.x_embedder = TimeSeries2EmbLinear(hidden_size=hidden_size, shape=(patch_num, patch_dim), dropout=dropout,)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = Linear(in_features=num_classes, out_features=hidden_size, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(num_classes)) if num_classes else None
        self.xo_embedder = TimeSeries2EmbLinear(hidden_size=hidden_size, shape=(in_len, in_channels), dropout=dropout,)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, num_experts, num_experts_per_tok, pretraining_tp, gamma, finetune) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear:
        nn.init.xavier_uniform_(self.x_embedder.processing[0].weight)
        nn.init.constant_(self.x_embedder.processing[0].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, xo):

        N,T,C = x.shape
        y = y.to(torch.float32)

        x =  x.view(N, T//self.patch_len, self.patch_len, self.in_channels).reshape(N, T//self.patch_len, self.in_channels*self.patch_len)

        x = self.x_embedder(x)                   
        t = self.t_embedder(t)                   
        y = self.y_embedder(y)                   
        xo = self.xo_embedder(self.emb_conv(xo)) 
        xo = torch.mean(xo, dim=1)               
        c = t + y                                
        for block in self.blocks:
            x = block(x, c, xo)                  
        x = self.final_layer(x, c)               

        x = x.view(N, T//self.patch_len, self.patch_len, self.in_channels).reshape(N, T, self.in_channels)

        return x