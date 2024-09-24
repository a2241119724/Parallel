from torch.nn import functional as F
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention
from models.ParallelEncoder_1 import ParallelEncoderLayer
from config.GlobalConfig import GlobalConfig

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff

class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=GlobalConfig.d_model, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, word_emb=None,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        # self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
        #                                           identity_map_reordering=identity_map_reordering,
        #                                           attention_module=attention_module,
        #                                           attention_module_kwargs=attention_module_kwargs)
        #                              for _ in range(N)])
        self.layers = nn.ModuleList([ParallelEncoderLayer(d_model, 1, word_emb, dropout, h) for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, image, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(regions, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1).bool()  # (b_s, 1, 1, seq_len)

        outs = []
        out = input
        for l in self.layers:
            # out = l(out, out, out, attention_mask, attention_weights)
            out, __sg_mask, sg, _sg_mask = l(image, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes)
            outs.append(out.unsqueeze(1))
        outs = torch.cat(outs, 1)
        return outs, attention_mask

class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, word_emb=None, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, word_emb=word_emb, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.fc1 = nn.Linear(d_in, self.d_model)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.layer_norm1 = nn.LayerNorm(self.d_model)

    def forward(self, enc_input, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes):
        out = F.relu(self.fc(enc_input))
        out = self.dropout1(out)
        out = self.layer_norm(out)
        regions = F.relu(self.fc1(regions))
        regions = self.dropout2(regions)
        regions = self.layer_norm1(regions)
        return super(MemoryAugmentedEncoder, self).forward(out, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes)
