import torch

from torch import nn
from .model import MultiHeadAttention, MultiHeadAttention_new
from .utils import get_activation_function
from config.GlobalConfig import GlobalConfig
from .ImageGCN import GCN
from .enums import DataType
from .model import FeedForward
from .model import GridAugmentedSelfAttention, GridAugmentedSelfAttention_new
from .model import MeshedMemoryMultiHeadAttention, MeshedMemoryMultiHeadAttention_new
from .model import MutilHeadXLinearAttention

class StackEncoderOne(nn.Module):
    def __init__(self, d_model:int, dropout, num_heads) -> None:
        super().__init__()
        # self.multi_head_attention_1 = MultiHeadAttention_new(d_model, 64, num_heads, True, dropout)
        # self.multi_head_attention_2 = MultiHeadAttention(d_model, 64, num_heads, True, dropout)
        self.multi_head_attention_1 = MeshedMemoryMultiHeadAttention_new(d_model, 64, num_heads, False, True, dropout)
        # self.multi_head_attention_2 = MeshedMemoryMultiHeadAttention_new(d_model, 64, num_heads, False, True, dropout)
        self.feed_forward1 = FeedForward(d_model, dropout)
        # self.feed_forward2 = FeedForward(d_model, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        # self.layer_norm3 = nn.LayerNorm(d_model)
        # self.layer_norm4 = nn.LayerNorm(d_model)

    def init(self):
        self.multi_head_attention_1.init()
        # self.multi_head_attention_2.init()
        self.feed_forward1.init()
        # self.feed_forward2.init()

    def forward(self, image, mask=None, pos_att=None):
        image = self.layer_norm1(self.multi_head_attention_1(image, mask=mask, pos_att=pos_att) / 10. + image)
        image = self.layer_norm2(self.feed_forward1(image) / 10. + image)
        # image = self.layer_norm3(self.multi_head_attention_2(image, mask=mask) / 10. + image)
        # image = self.layer_norm4(self.feed_forward2(image) / 10. + image)
        return image

class StackEncoderTwo(nn.Module):
    def __init__(self, d_model:int, word_emb) -> None:
        super().__init__()
        self.gcn_conv = GCN(d_model, word_emb)
        self.layernorm = nn.LayerNorm(d_model)

    def init(self):
        self.gcn_conv.init()

    def forward(self, image_id, enti2attr, sub2obj2rela, sg, sg_mask, _enti2attr, _sub2obj2rela, boxes):
        # 1
        # edge_index = self.gcn_conv.get_edge_index(caption).to(self.device)
        # for i in range(self.batch_size):
        #     _edge_index = edge_index[i, :, :edge_index[i,0,-1]].contiguous()
        #     caption2vector_new[i, :, :] = self.layer_norm(self.gcn_conv(caption2vector[i, :, :].contiguous(), 
        #         _edge_index) + caption2vector[i, :, :].contiguous())
        # 2
        # caption2vector = self.gcn_conv(caption2vector,caption)
        # 3
        sg, sg_mask, _enti2attr, _sub2obj2rela, obj_obj = self.gcn_conv(image_id, enti2attr, sub2obj2rela, sg, sg_mask, _enti2attr, _sub2obj2rela)
        sg = self.layernorm(sg)
        return sg, sg_mask, _enti2attr, _sub2obj2rela, obj_obj


class ParallelEncoderLayer(nn.Module):
    def __init__(self, d_model:int, N_enc:int, word_emb, dropout, num_heads) -> None:
        super().__init__()
        self.word_emb = word_emb
        self.stack_encoder_one_layers = nn.ModuleList([StackEncoderOne(d_model, dropout, num_heads) for _ in range(N_enc)])
        # self.stack_encoder_one_layers_1 = nn.ModuleList([StackEncoderOne(d_model, dropout, num_heads) for _ in range(N_enc)])
        self.stack_encoder_two_layers = nn.ModuleList([StackEncoderTwo(d_model, self.word_emb) for _ in range(N_enc)])
        self.layer_norm = nn.LayerNorm(d_model)
        # self.x_linear_attention = MutilHeadXLinearAttention(d_model, 64, num_heads)
        self.linear = nn.Sequential(
            nn.Linear(4, d_model),
            get_activation_function(),
            nn.Dropout(GlobalConfig.dropout)
        )

    def init(self):
        nn.init.xavier_uniform_(self.linear[0].weight)
        nn.init.constant_(self.linear[0].bias, 0)

        for l in self.stack_encoder_one_layers:
            l.init()
        # for l in self.stack_encoder_one_layers_1:
        #     l.init()
        for l in self.stack_encoder_two_layers:
            l.init()
        # self.x_linear_attention.init()

    def forward(self, image, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes, pos_att=None):
        for l in self.stack_encoder_one_layers:
            image = l(image, pos_att=pos_att)
        # sg_mask = None
        # for l in self.stack_encoder_one_layers_1:
        #     regions = l(regions, sg_mask[1])
        # image = regions
        # sg_mask = sg_mask[1]
        sg, _enti2attr, _sub2obj2rela = None, None, None
        for l in self.stack_encoder_two_layers:
            sg, sg_mask, _enti2attr, _sub2obj2rela, obj_obj = l(image_id, enti2attr, sub2obj2rela, sg, sg_mask, _enti2attr, _sub2obj2rela, boxes)
        # image = self.layer_norm(sg)
        # 1
        sg_mask = torch.cat([torch.zeros(image.shape[:2], dtype=torch.bool).to(sg_mask.device), sg_mask], dim=1)
        image = self.layer_norm(torch.cat((image, sg), dim=1))
        ## 2
        # image = self.layer_norm(self.x_linear_attention(image, sg, sg_mask) + image)
        # image = self.layer_norm(self.x_linear_attention(image, image.mean(-2).unsqueeze(-2)) + image)
        # sg_mask = None
        #
        # image = self.layer_norm(torch.cat([self.x_linear_attention(image, regions, _sg_mask[1]), obj_obj + self.linear(boxes), sg], dim=1))
        # image = self.layer_norm(torch.cat([self.x_linear_attention(image, regions, _sg_mask[1]), sg], dim=1))
        # return image, sg_mask, obj_obj + self.linear(boxes)
        return image, sg_mask, obj_obj

class ParallelEncoder(nn.Module):
    def __init__(self, d_model:int, N_enc:int, word_emb, dropout, num_heads) -> None:
        super().__init__()
        self.word_emb = word_emb
        self.layers = nn.ModuleList([ParallelEncoderLayer(d_model, N_enc, self.word_emb, dropout, num_heads) for _ in range(1)])
        # self.grid_augmented_self_attention = GridAugmentedSelfAttention_new(num_heads)

    def init(self):
        for l in self.layers:
            l.init()
        # self.grid_augmented_self_attention.init()

    def forward(self, image, image_id=None, enti2attr=None, sub2obj2rela=None, sg_mask=None, regions=None, boxes=None):
        pos_att = None
        # pos_att = self.grid_augmented_self_attention(image)
        for l in self.layers:
            image, sg_mask, obj_obj = l(image, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes, pos_att)
        return image, sg_mask, obj_obj
