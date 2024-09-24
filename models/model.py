import torch
import math
import numpy as np
import torch.nn.functional as F

from torch import nn
from .utils import get_activation_function
from config.GlobalConfig import GlobalConfig
from .utils import get_grids_pos
from torch.cuda.amp import autocast

class FeedForward(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            get_activation_function("relu"),
            nn.Dropout(dropout)
        )
        self.w_2 = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            get_activation_function(),
            nn.Dropout(dropout)
        )

    def init(self):
        nn.init.xavier_uniform_(self.w_1[0].weight)
        nn.init.xavier_uniform_(self.w_2[0].weight)

        nn.init.constant_(self.w_1[0].bias, 0)
        nn.init.constant_(self.w_2[0].bias, 0)

    def forward(self, x):
        x = self.w_2(self.w_1(x))
        return x

class SelfAttention(nn.Module):
    
    def __init__(self, d_model:int, d_kqv:int, num_heads:int, dropout:float, bias:bool=True):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_kqv = d_kqv

    def init(self):
        pass

    def forward(self, query, key, value, mask=None):
        q,k,v = query,key,value
        attention = torch.matmul(q,k.transpose(-1,-2)) * (self.d_kqv ** -0.5)
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(-2), 1e-9)
        attention = F.softmax(attention, -1)
        res = torch.matmul(attention, v)
        return res

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_kqv, num_heads, bias, dropout):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_kqv = d_kqv
        self.w_q = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.w_k = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.w_v = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.self_attention = SelfAttention(d_model, d_kqv, num_heads, bias, dropout)
        self.w_o = nn.Sequential(
            nn.Linear(d_kqv * num_heads, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )

    def init(self):
        nn.init.xavier_uniform_(self.w_k[0].weight)
        nn.init.xavier_uniform_(self.w_v[0].weight)
        nn.init.xavier_uniform_(self.w_o[0].weight)
        nn.init.xavier_uniform_(self.w_q[0].weight)

        nn.init.constant_(self.w_q[0].bias, 0)
        nn.init.constant_(self.w_k[0].bias, 0)
        nn.init.constant_(self.w_v[0].bias, 0)
        nn.init.constant_(self.w_o[0].bias, 0)

        self.self_attention.init()

    def forward(self, x, y=None, mask=None):
        ''' 3d '''
        y = y if y is not None else x
        batch_size, x_1 = x.shape[:2]
        batch_size, y_1 = y.shape[:2]
        qs = self.w_q(x).view(batch_size, x_1, self.num_heads, self.d_kqv)
        ks = self.w_k(y).view(batch_size, y_1, self.num_heads, self.d_kqv)
        vs = self.w_v(y).view(batch_size, y_1, self.num_heads, self.d_kqv)
        out = torch.zeros((batch_size, x_1, self.num_heads, self.d_kqv)).to(x.device)
        for i in range(self.num_heads):
            out[:, :, i, :] = self.self_attention(qs[:, :, i, :].contiguous(), 
                ks[:, :, i, :].contiguous(), vs[:, :, i, :].contiguous(), mask)
        out = self.w_o(out.view(batch_size, x_1, self.num_heads * self.d_kqv))
        return out

class MultiHeadAttention_new(nn.Module):
    def __init__(self, d_model, d_kqv, num_heads, bias, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_kqv
        self.d_v = d_kqv
        self.h = num_heads

        act = ""
        if d_model > d_kqv * num_heads:
            act = "relu"

        self.w_q = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(act),
            nn.Dropout(p=dropout),
        )
        self.w_k = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(act),
            nn.Dropout(p=dropout),
        )
        self.w_v = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(act),
            nn.Dropout(p=dropout),
        )
        self.w_o = nn.Sequential(
            nn.Linear(d_kqv * num_heads, d_model, bias=bias),
            get_activation_function(),
            # if d_model == d_kqv * num_heads:
            nn.Dropout(p=dropout),
        )

    def init(self):
        nn.init.xavier_uniform_(self.w_q[0].weight)
        nn.init.xavier_uniform_(self.w_k[0].weight)
        nn.init.xavier_uniform_(self.w_v[0].weight)
        nn.init.xavier_uniform_(self.w_o[0].weight)

        nn.init.constant_(self.w_q[0].bias, 0)
        nn.init.constant_(self.w_k[0].bias, 0)
        nn.init.constant_(self.w_v[0].bias, 0)
        nn.init.constant_(self.w_o[0].bias, 0)
    
    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        qs, ks, vs = x, y, y
        b_s, nq = qs.shape[:2]
        nk = ks.shape[1]
        q = self.w_q(qs).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.w_k(ks).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.w_v(vs).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if mask is not None:
            att = att.masked_fill(mask.bool(), -1e9)

        att = F.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.w_o(out)
        return out

class Attention(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout),
        )
        self.linear1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout),
        )
        self.tanh = nn.Tanh()
        self.linear2 = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout),
        )
        self.d_model = d_model

    def init(self):
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.xavier_uniform_(self.linear1[0].weight)
        nn.init.xavier_uniform_(self.linear2[0].weight)

        nn.init.constant_(self.linear[0].bias, 0)
        nn.init.constant_(self.linear1[0].bias, 0)
        # nn.init.constant_(self.linear2[0].bias, 0)

    def forward(self, x, y=None, mask=None):
        _y = self.linear1(y)
        y = _y.unsqueeze(1).expand_as(x)
        y = self.tanh(self.linear(x) + y)
        y = self.linear2(y).transpose(-1,-2)
        if mask is not None:
            y = y.masked_fill(mask.unsqueeze(-2), -1e9)
        att = F.softmax(y, -1)
        x = torch.matmul(att, x)
        x = x.squeeze(1)
        return x, _y

class XLinearAttention(nn.Module):
    def __init__(self, d_model:int, bias:bool=True):
        super(XLinearAttention, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.squeeze_excitation = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
            nn.Sigmoid()
        )
        #
        self.linear2 = nn.Sequential(
            nn.Linear(d_model, 1),
            get_activation_function("relu"),
            nn.Dropout(p=GlobalConfig.dropout)
        )

    def init(self):
        nn.init.xavier_uniform_(self.linear1[0].weight)
        nn.init.xavier_uniform_(self.linear2[0].weight)
        nn.init.xavier_uniform_(self.squeeze_excitation[0].weight)

        nn.init.constant_(self.linear1[0].bias, 0)
        nn.init.constant_(self.linear2[0].bias, 0)
        nn.init.constant_(self.squeeze_excitation[0].bias, 0)

    def forward(self, qs_1, qs_2, ks, vs, sg_mask=None):
        val_2 = vs.shape[-2]
        que_2 = qs_1.shape[-2]
        bl_1 = qs_1.unsqueeze(-2).repeat(1,1,val_2,1).reshape(-1, val_2, self.d_model) \
              * ks.unsqueeze(-2).repeat(1,que_2,1,1).reshape(-1, val_2, self.d_model)
        bl_2 = vs.unsqueeze(-2).repeat(1,que_2,1,1).reshape(-1, val_2, self.d_model)
        # bl_2 = qs_2.unsqueeze(-2).repeat(1,1,val_2,1).reshape(-1, val_2, self.d_model) \
        #       * vs.unsqueeze(-2).repeat(1,que_2,1,1).reshape(-1, val_2, self.d_model)
        # bl_2 = qs_2.unsqueeze(-2).repeat(1,1,val_2,1).reshape(-1, val_2, self.d_model) \
        #     + vs.unsqueeze(-2).repeat(1,que_2,1,1).reshape(-1, val_2, self.d_model)
        ##
        bl_1 = self.linear1(bl_1)
        bl_1_2 = self.linear2(bl_1)
        if sg_mask is not None:
            _sg_mask = sg_mask.unsqueeze(1).repeat(1, que_2, 1).reshape(-1, val_2)
            __sg_mask = _sg_mask.unsqueeze(-1).repeat(1, 1, bl_1.shape[-1])
            bl_1 = bl_1.masked_fill(__sg_mask, 0.)
            bl_1_1 = bl_1.sum(dim=1) / (~_sg_mask).sum(dim=1).unsqueeze(-1)
            bl_1_1 = bl_1_1.unsqueeze(-2)
        else:
            # bl_1_1 = self.adaptive_avg_pool_1d(bl_1.transpose(-1, -2)).transpose(-1, -2)
            bl_1_1 = bl_1.mean(-2).unsqueeze(-2)
        bl_1_1 = self.squeeze_excitation(bl_1_1)
        if sg_mask is not None:
            # mask
            # sg_mask = sg_mask.unsqueeze(2).repeat(1, 1, val_2).unsqueeze(-1).reshape(-1, val_2, 1)
            sg_mask = sg_mask.unsqueeze(1).repeat(1, que_2, 1).unsqueeze(-1).reshape(-1, val_2, 1)
            bl_1_2 = bl_1_2.masked_fill(sg_mask, -1e9)
        bl_1_2 = F.softmax(bl_1_2, -2)
        #
        bl_2 = torch.matmul(bl_1_2.transpose(-1, -2), bl_2)
        bl = bl_1_1 * bl_2
        bl = bl.reshape(-1, que_2, self.d_model)
        return bl

class MutilHeadXLinearAttention(nn.Module):
    
    def __init__(self, d_model:int, d_kqv:int, num_heads:int, bias:bool=True):
        super(MutilHeadXLinearAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_kqv = d_kqv
        self.x_linear_attention = XLinearAttention(d_kqv, bias)
        self.w_q_1 = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads),
            get_activation_function("elu"),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_q_2 = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads),
            get_activation_function("elu"),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_k = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads),
            get_activation_function("elu"),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_v = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads),
            get_activation_function("elu"),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_o = nn.Sequential(
            nn.Linear(d_kqv * num_heads, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )

    def init(self):
        nn.init.xavier_uniform_(self.w_q_1[0].weight)
        nn.init.xavier_uniform_(self.w_q_2[0].weight)
        nn.init.xavier_uniform_(self.w_k[0].weight)
        nn.init.xavier_uniform_(self.w_v[0].weight)
        nn.init.xavier_uniform_(self.w_o[0].weight)
        
        nn.init.constant_(self.w_q_1[0].bias, 0)
        nn.init.constant_(self.w_q_2[0].bias, 0)
        nn.init.constant_(self.w_k[0].bias, 0)
        nn.init.constant_(self.w_v[0].bias, 0)
        nn.init.constant_(self.w_o[0].bias, 0)
        
        self.x_linear_attention.init()

    def forward(self, query, value, sg_mask=None):
        batch_size, val_2 = value.shape[:2]
        que_2 = query.shape[-2]
        qs_1 = self.w_q_1(query).view(batch_size, que_2, self.num_heads, self.d_kqv)
        qs_2 = self.w_q_2(query).view(batch_size, que_2, self.num_heads, self.d_kqv)
        ks = self.w_k(value).view(batch_size, val_2, self.num_heads, self.d_kqv)
        vs = self.w_v(value).view(batch_size, val_2, self.num_heads, self.d_kqv)
        out = torch.zeros((batch_size, que_2, self.num_heads, self.d_kqv)).to(query.device)
        for i in range(self.num_heads):
            out[:, :, i, :] = self.x_linear_attention(qs_1[:, :, i, :].contiguous(), 
                qs_2[:, :, i, :].contiguous(), ks[:, :, i, :].contiguous(), 
                vs[:, :, i, :].contiguous(), sg_mask)
        out = self.w_o(out.view(batch_size, que_2, self.num_heads * self.d_kqv))
        return out

class MeshedMemoryMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_kqv, num_heads, is_memory=False, bias=True, dropout=0.):
        super(MeshedMemoryMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_kqv = d_kqv
        
        self.w_q = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.w_k = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.w_v = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.self_attention = SelfAttention(d_model, d_kqv, num_heads, bias, dropout)
        self.w_o = nn.Sequential(
            nn.Linear(d_kqv * num_heads, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.is_memory = is_memory
        if is_memory:
            self.memory_k = nn.Parameter(torch.randn(1, 40, num_heads, d_kqv))
            self.memory_v = nn.Parameter(torch.randn(1, 40, num_heads, d_kqv))

    def init(self):
        if self.is_memory:
            nn.init.xavier_uniform_(self.memory_k)
            nn.init.xavier_uniform_(self.memory_v)
        nn.init.xavier_uniform_(self.w_q[0].weight)
        nn.init.xavier_uniform_(self.w_k[0].weight)
        nn.init.xavier_uniform_(self.w_v[0].weight)
        nn.init.xavier_uniform_(self.w_o[0].weight)

        nn.init.constant_(self.w_q[0].bias, 0)
        nn.init.constant_(self.w_k[0].bias, 0)
        nn.init.constant_(self.w_v[0].bias, 0)
        nn.init.constant_(self.w_o[0].bias, 0)

        self.self_attention.init()

    def forward(self, x, y=None, mask=None):
        ''' 3d '''
        y = y if y is not None else x
        batch_size, x_1 = x.shape[:2]
        batch_size, y_1 = y.shape[:2]
        qs = self.w_q(x).view(batch_size, x_1, self.num_heads, self.d_kqv)
        ks = self.w_k(y).view(batch_size, y_1, self.num_heads, self.d_kqv)
        vs = self.w_v(y).view(batch_size, y_1, self.num_heads, self.d_kqv)
        if self.is_memory:
            ks = torch.cat([ks, self.memory_k.repeat(batch_size, 1, 1, 1)], dim=1)
            vs = torch.cat([vs, self.memory_v.repeat(batch_size, 1, 1, 1)], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.ones((batch_size, self.memory_k.shape[1]), dtype=torch.bool).to(x.device)], dim=1)
        out = torch.zeros((batch_size, x_1, self.num_heads, self.d_kqv)).to(x.device)
        for i in range(self.num_heads):
            out[:, :, i, :] = self.self_attention(qs[:, :, i, :].contiguous(), 
                ks[:, :, i, :].contiguous(), vs[:, :, i, :].contiguous(), mask)
        out = self.w_o(out.view(batch_size, x_1, self.num_heads * self.d_kqv))
        return out
    
class MeshedMemoryMultiHeadAttention_new(nn.Module):
    def __init__(self, d_model, d_kqv, num_heads, is_memory=False, bias=True, dropout=0.):
        super(MeshedMemoryMultiHeadAttention_new, self).__init__()
        self.h = num_heads
        self.d_v = d_kqv
        self.d_k = d_kqv

        self.w_q = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout),
        )
        self.w_k = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout),
        )
        self.w_v = nn.Sequential(
            nn.Linear(d_model, d_kqv * num_heads, bias=bias),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout),
        )
        self.self_attention = SelfAttention(d_model, d_kqv, num_heads, bias, dropout)
        self.w_o = nn.Sequential(
            nn.Linear(d_kqv * num_heads, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout),
        )
        self.is_memory = is_memory
        if is_memory:
            self.slots = 40
            self.memory_k = nn.Parameter(torch.FloatTensor(1, self.slots, num_heads * d_kqv))
            self.memory_v = nn.Parameter(torch.FloatTensor(1, self.slots, num_heads * d_kqv))

    def init(self):
        if self.is_memory:
            # nn.init.xavier_uniform_(self.memory_k)
            # nn.init.xavier_uniform_(self.memory_v)
            nn.init.normal_(self.memory_k, 0, 1 / self.d_k)
            nn.init.normal_(self.memory_v, 0, 1 / self.slots)
        nn.init.xavier_uniform_(self.w_q[0].weight)
        nn.init.xavier_uniform_(self.w_k[0].weight)
        nn.init.xavier_uniform_(self.w_v[0].weight)
        nn.init.xavier_uniform_(self.w_o[0].weight)

        nn.init.constant_(self.w_q[0].bias, 0)
        nn.init.constant_(self.w_k[0].bias, 0)
        nn.init.constant_(self.w_v[0].bias, 0)
        nn.init.constant_(self.w_o[0].bias, 0)

        self.self_attention.init()

    def forward(self, x, y=None, mask=None, pos_att=None, decoder_mask=False):
        y = y if y is not None else x
        qs, ks, vs = x, y, y
        b_s, nq = qs.shape[:2]
        nk = ks.shape[1]
        # if pos_att is not None:
        #     pos_att = pos_att.unsqueeze(0).repeat(b_s, 1, 1)
        q = self.w_q(qs).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        if self.is_memory:
            memory_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.slots, self.h * self.d_k)
            memory_v = np.sqrt(self.slots) * self.m_v.expand(b_s, self.slots, self.h * self.d_v)
            ks = torch.cat([self.w_k(ks), memory_k], dim=1).view(b_s, nk + self.slots, self.h, self.d_k).permute(0, 2, 3, 1)
            vs = torch.cat([self.w_v(vs), memory_v], dim=1).view(b_s, nk + self.slots, self.h, self.d_v).permute(0, 2, 1, 3)
            if pos_att is not None:
                pos_att = torch.cat([pos_att, torch.zeros((b_s, nq, self.slots)).to(x.device)], dim=-1)
        else:
            k = self.w_k(ks).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
            v = self.w_v(vs).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
            
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if pos_att is not None:
            att = att + pos_att.unsqueeze(1).repeat(1, self.h, 1, 1)
        radio = torch.ones_like(att) * 0.3
        att = att + torch.bernoulli(radio) * -1e9
        if mask is not None:
            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(mask.unsqueeze(1).unsqueeze(1).repeat(1, self.h, nq, 1), -1e9)
        if decoder_mask:
            _mask = torch.triu(torch.ones((b_s, self.h, GlobalConfig.max_seq_len, GlobalConfig.max_seq_len), device=att.device), diagonal=1).bool()
            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(_mask, -1e9)

        att = F.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.w_o(out)
        return out

class GridAugmentedSelfAttention(nn.Module):
    def __init__(self, d_model, bias=True) -> None:
        super(GridAugmentedSelfAttention, self).__init__()
        self.d_model = d_model

        self.w_q = nn.Sequential(
            nn.Linear(d_model, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_k = nn.Sequential(
            nn.Linear(d_model, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_v = nn.Sequential(
            nn.Linear(d_model, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )

        self.linear = nn.Sequential(
            # nn.Linear(2, d_model, bias=False),
            nn.Linear(4, d_model, bias=False),
            get_activation_function(),
            # nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_g = nn.Parameter(torch.randn((144, d_model, 1)))
        self.is_one = True
        self.position = None

    def init(self):
        nn.init.xavier_uniform_(self.w_g)
        nn.init.xavier_uniform_(self.w_q[0].weight)
        nn.init.xavier_uniform_(self.w_k[0].weight)
        nn.init.xavier_uniform_(self.w_v[0].weight)
        nn.init.xavier_uniform_(self.linear[0].weight)

        nn.init.constant_(self.w_q[0].bias, 0)
        nn.init.constant_(self.w_k[0].bias, 0)
        nn.init.constant_(self.w_v[0].bias, 0)
        # nn.init.constant_(self.linear[0].bias, 0)
    
    def forward(self, query, key, value, mask=None):
        b_s = query.shape[0]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        attention = torch.matmul(q,k.transpose(-1,-2)) * (self.d_model  ** -0.5)
        #
        rela_pos = self.linear(self.get_relative_position(value))
        rela_pos = torch.matmul(rela_pos, self.w_g)
        rela_pos = F.relu(rela_pos).squeeze(-1)
        attention = attention + rela_pos.unsqueeze(0).expand(b_s, -1, -1)
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(3), 1e-9)
        attention = F.softmax(attention, -1)
        res = torch.matmul(attention, v)
        return res

    def get_relative_position(self, grid):
        if not self.is_one:
            return self.position
        total = grid.shape[1]
        x_min, y_min, x_max, y_max = get_grids_pos(int(math.sqrt(total)), grid.device)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5

        delta_x = cx - cx.view(1, -1)
        angle_x = torch.sign(delta_x) * 0.5
        delta_x = torch.abs(torch.log(torch.abs(delta_x) + 0.5))

        delta_y = cy - cy.view(1, -1)
        angle_y = torch.sign(delta_y) * 0.5
        delta_y = torch.abs(torch.log(torch.abs(delta_y) + 0.5))

        # position = torch.cat((delta_x, delta_y), 0).permute(1, 2, 0)
        position = torch.cat((delta_x, delta_y, angle_x, angle_y), 0).permute(1, 2, 0)
        self.position = position
        self.is_one = not self.is_one
        return position

class RegionAugmentedSelfAttention(nn.Module):
    def __init__(self, d_model, bias=True) -> None:
        super(RegionAugmentedSelfAttention, self).__init__()
        self.scale = d_model ** -0.5
        self.w_q = nn.Sequential(
            nn.Linear(d_model, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_k = nn.Sequential(
            nn.Linear(d_model, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_v = nn.Sequential(
            nn.Linear(d_model, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )

        self.linear = nn.Sequential(
            nn.Linear(4, d_model, bias=bias),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.w_g = nn.Parameter(torch.randn((144, d_model, 1)))
        self.relu = get_activation_function("relu")

    def init(self):
        nn.init.xavier_uniform_(self.w_g)
        nn.init.xavier_uniform_(self.w_q[0].weight)
        nn.init.xavier_uniform_(self.w_k[0].weight)
        nn.init.xavier_uniform_(self.w_v[0].weight)
        nn.init.xavier_uniform_(self.linear[0].weight)

        nn.init.constant_(self.w_q[0].bias, 0)
        nn.init.constant_(self.w_k[0].bias, 0)
        nn.init.constant_(self.w_v[0].bias, 0)
        nn.init.constant_(self.linear[0].bias, 0)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        attention = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        #
        rela_pos = self.linear(self.get_relative_position(value))
        rela_pos = torch.matmul(rela_pos, self.w_g)
        rela_pos = self.relu(rela_pos).squeeze(-1)
        attention = attention + rela_pos.unsqueeze(0).expand(batch_size, -1, -1)
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(3), 1e-10)
        attention = F.softmax(attention, -1)
        res = torch.matmul(attention, v)
        return res

    def get_relative_position(self, grid):
        total = grid.shape[1]
        x_min, y_min, x_max, y_max = get_grids_pos(int(math.sqrt(total)), grid.device)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.

        # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
        delta_x = cx - cx.view(1, -1)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-10)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(1, -1)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-10)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(1, -1))
        delta_h = torch.log(h / h.view(1, -1))

        matrix_size = delta_h.size()
        delta_x = delta_x.view(matrix_size[1], matrix_size[2], 1)
        delta_y = delta_y.view(matrix_size[1], matrix_size[2], 1)
        delta_w = delta_w.view(matrix_size[1], matrix_size[2], 1)
        delta_h = delta_h.view(matrix_size[1], matrix_size[2], 1)

        position = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
        return position

class GridAugmentedSelfAttention_new(nn.Module):
    def __init__(self, num_heads) -> None:
        super(GridAugmentedSelfAttention_new, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(64, 64 * 4),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 4, 64),
            get_activation_function(),
            nn.Dropout(p=GlobalConfig.dropout)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(64, 1, bias=False),
            get_activation_function("relu"),
            # nn.Dropout(p=GlobalConfig.dropout)
        )
        self.is_one = True
        self.position = None

    def init(self):
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.xavier_uniform_(self.linear1[0].weight)
        nn.init.xavier_uniform_(self.linear2[0].weight)

        nn.init.constant_(self.linear[0].bias, 0)
        nn.init.constant_(self.linear1[0].bias, 0)
        # nn.init.constant_(self.linear2[0].bias, 0)
    
    def forward(self, grid):
        b_s, total = grid.shape[:2]
        rela_pos = self.get_relative_position(total, grid.device).unsqueeze(0).repeat(b_s, 1, 1, 1)
        rela_pos = self.linear(rela_pos)
        rela_pos = self.linear1(rela_pos)
        rela_pos = self.linear2(rela_pos).squeeze(-1)
        return rela_pos / 10.

    def get_relative_position(self, total, device):
        if not self.is_one:
            return self.position
        x_min, y_min, x_max, y_max = get_grids_pos(int(math.sqrt(total)), device)

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.

        delta_x = cx - cx.view(1, -1)
        angle_x = torch.sign(delta_x)
        delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
        delta_x = torch.log(delta_x)

        delta_y = cy - cy.view(1, -1)
        angle_y = torch.sign(delta_y)
        delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
        delta_y = torch.log(delta_y)

        delta_w = torch.log(w / w.view(1, -1))
        delta_h = torch.log(h / h.view(1, -1))

        matrix_size = delta_h.size()
        # position = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
        position = torch.cat((delta_x, delta_y, angle_x, angle_y), 0).permute(1, 2, 0)
        position = F.layer_norm(position, (position.shape[-1],))
        position = position.view(matrix_size[1], matrix_size[2], 4, -1)
        position = 100. * position

        feat_range = torch.arange(64 / 8).to(device)
        dim_mat = feat_range / (64 / 8)
        dim_mat = 1. / (torch.pow(1000, dim_mat))
        dim_mat = dim_mat.view(1, 1, 1, -1)

        mul_mat = position * dim_mat
        mul_mat = mul_mat.contiguous().view(matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
        self.position = embedding
        self.is_one = not self.is_one
        return embedding