import torch
import math
import torch.nn.functional as F

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
from .utils import get_grids_pos
from .utils import sinusoid_encoding_table

class ParallelFeatureFusion(nn.Module):
    def __init__(self, d_model:int, dropout, num_heads) -> None:
        super().__init__()
        self.w_att = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            # nn.Dropout(dropout)
        )
        self.w_extra_att = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            # nn.Dropout(dropout)
        )
        self.w_fusion_att = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            get_activation_function(),
            # nn.Dropout(dropout)
        )
        self.w_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            get_activation_function("relu"),
            nn.Dropout(dropout)
        )

    def init(self):
        nn.init.xavier_uniform_(self.w_att[0].weight)
        nn.init.xavier_uniform_(self.w_extra_att[0].weight)
        nn.init.xavier_uniform_(self.w_fusion_att[0].weight)
        nn.init.xavier_uniform_(self.w_fusion[0].weight)

        nn.init.constant_(self.w_att[0].bias, 0.)
        nn.init.constant_(self.w_extra_att[0].bias, 0.)
        # nn.init.constant_(self.w_fusion_att[0].bias, 0.)
        nn.init.constant_(self.w_fusion[0].bias, 0.)

    def forward(self, region, grid, extra_att, mask):
        g_2 = grid.shape[-2]
        r_2 = region.shape[-2]
        grid = grid.unsqueeze(1).repeat(1, r_2, 1, 1)
        _region = region.unsqueeze(2).repeat(1, 1, g_2, 1)
        att = F.sigmoid(self.w_att(_region * grid))
        grid = att * grid
        size = list(extra_att.shape[:2])
        size.append(-1)
        att = self.w_fusion_att(att + self.w_extra_att(extra_att))
        # att = att.masked_fill(mask.unsqueeze(-1), -1e9)
        att = att * mask.unsqueeze(-1)
        att = F.softmax(att, dim=-2)
        grid = (torch.matmul(att.transpose(-1, -2), grid)).squeeze(-2)
        image = torch.cat([region, grid], dim=-1)
        image = self.w_fusion(image)
        return image

class ParallelEncoderLayer(nn.Module):
    def __init__(self, d_model:int, N_enc:int, word_emb, dropout, num_heads) -> None:
        super().__init__()
        self.word_emb = word_emb
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.w_grid = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            # nn.Dropout(dropout)
        )
        self.w_region = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            # nn.Dropout(dropout)
        )
        self.feature_fusion = ParallelFeatureFusion(d_model, dropout, num_heads)
        self.feature_fusion_1 = ParallelFeatureFusion(d_model, dropout, num_heads)
        self.w_att = nn.Sequential(
            nn.Linear(49, 49),
            get_activation_function("relu"),
            # nn.Dropout(dropout)
        )
        self.w_att_1 = nn.Sequential(
            nn.Linear(30, 30),
            get_activation_function("relu"),
            # nn.Dropout(dropout)
        )
        self.multi_head_attention_1 = MeshedMemoryMultiHeadAttention_new(d_model, 64, num_heads, False, True, dropout)
        self.multi_head_attention_2 = MeshedMemoryMultiHeadAttention_new(d_model, 64, num_heads, False, True, dropout)
        self.multi_head_attention_3 = MeshedMemoryMultiHeadAttention_new(d_model, 64, num_heads, False, True, dropout)
        self.feed_forward_1 = FeedForward(d_model, dropout)
        self.feed_forward_2 = FeedForward(d_model, dropout)
        self.feed_forward_3 = FeedForward(d_model, dropout)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(100 + 1, d_model, 0), freeze=True)
        self.gcn = GCN(d_model, word_emb)

    def init(self):
        nn.init.xavier_uniform_(self.w_grid[0].weight)
        nn.init.xavier_uniform_(self.w_region[0].weight)
        nn.init.xavier_uniform_(self.w_att[0].weight)
        nn.init.xavier_uniform_(self.w_att[1].weight)
        nn.init.xavier_uniform_(self.w_att_1[0].weight)
        nn.init.xavier_uniform_(self.w_att_1[1].weight)

        nn.init.constant_(self.w_grid[0].bias, 0.)
        nn.init.constant_(self.w_region[0].bias, 0.)
        nn.init.constant_(self.w_att[0].bias, 0.)
        nn.init.constant_(self.w_att[1].bias, 0.)
        nn.init.constant_(self.w_att_1[0].bias, 0.)
        nn.init.constant_(self.w_att_1[1].bias, 0.)
        
        self.feature_fusion.init()
        self.feature_fusion_1.init()
        self.multi_head_attention_1.init()
        self.multi_head_attention_2.init()
        self.multi_head_attention_3.init()
        self.feed_forward_1.init()
        self.feed_forward_2.init()
        self.feed_forward_3.init()
        self.gcn.init()

    def forward(self, grid, image_id, enti2attr, sub2obj2rela, sg_mask, region, boxes, pos_att=None):
        sg, _sg_mask = self.gcn(enti2attr, sub2obj2rela, sg_mask)
        sg = F.layer_norm(self.multi_head_attention_3(sg, mask=_sg_mask) + sg, (sg.shape[-1],))
        sg = F.layer_norm(self.feed_forward_3(sg) + sg, (sg.shape[-1],))
        ##
        b_s = boxes.shape[0]
        g_2 = grid.shape[1]
        r_2 = region.shape[1]
        # 位置编码(x_min, y_min, x_max, y_max)
        grid_pos = torch.cat(get_grids_pos(int(math.sqrt(grid.shape[-2])), grid.device), dim=-1).repeat(b_s, 1, 1)
        boxes[:, :, 0::2] = boxes[:, :, 0::2] / 800.
        boxes[:, :, 1::2] = boxes[:, :, 1::2] / 600.
        region_pos = boxes # (600, 800)
        region_area = region_pos[:, :, 2:] - region_pos[:, :, :2]
        region_area = region_area[:, :, 0] * region_area[:, :, 1]
        grid_area = grid_pos[:, 0, 2:] - grid_pos[:, 0, :2]
        grid_area = grid_area[:, 0] * grid_area[:, 1]
        #
        _region_pos = region_pos.unsqueeze(2).repeat(1, 1, g_2, 1)
        _grid_pos = grid_pos.unsqueeze(1).repeat(1, r_2, 1, 1)
        area = torch.stack([_region_pos, _grid_pos], dim=-2)
        # 中心位置距离
        area_center = area.reshape(b_s, r_2, g_2, 2, 2, 2).mean(dim=-2)
        area_center_wh = torch.abs(area_center[:, :, :, 0, :] - area_center[:, :, :, 1, :])
        area_center_len = area_center_wh[:, :, :, 0] * area_center_wh[:, :, :, 1]
        #
        max_area = area[:, :, :, :, :2].max(dim=-2).values
        min_area = area[:, :, :, :, 2:].min(dim=-2).values
        area = torch.cat([max_area, min_area], dim=-1) # 重叠区域的左上右下坐标
        # 重叠的面积
        zero_mask = (max_area >= min_area).int().sum(-1).bool()
        wh_area = torch.abs(min_area - max_area) # 重叠部分的宽高
        cover_area = wh_area[:, :, :, 0] * wh_area[:, :, :, 1]
        cover_area = cover_area.masked_fill(zero_mask, 0.)
        # 重叠部分所占区域的百分比
        cover_percent = cover_area / (region_area.unsqueeze(2).repeat(1, 1, g_2) + 1e-9)
        #
        _extra_att = cover_percent + cover_area / cover_area.max() - torch.pow(area_center_len, 2) / area_center_len.max()
        extra_att = self.pos_emb((F.relu(_extra_att) / 3 * 100).int())
        ##
        # region2grid_num = g_2
        cover_percent = cover_percent.masked_fill(sg_mask[1].unsqueeze(-1), 0.)
        region2grid_num = (cover_percent > 0.).int().sum(-1)
        grid2region_num = (cover_percent > 0.).int().sum(-2)
        # grid2region_num = (~sg_mask[1]).int().sum(-1).unsqueeze(-1).repeat(1, g_2)
        _mask = torch.arange(0, g_2, device=region2grid_num.device).unsqueeze(0).unsqueeze(0).repeat(b_s, r_2, 1)
        _mask_1 = torch.arange(0, r_2, device=region2grid_num.device).unsqueeze(0).unsqueeze(0).repeat(b_s, g_2, 1)
        _mask = _mask >= region2grid_num.unsqueeze(-1)
        _mask_1 = _mask_1 >= grid2region_num.unsqueeze(-1)
        _region = self.w_region(region)
        _grid = self.w_grid(grid)
        _att = torch.sigmoid(torch.matmul(_region, _grid.transpose(-2, -1)))
        att_1 = self.w_att_1(_att.transpose(-1, -2))
        rate_1 = _extra_att.max(-2)[0] / (att_1.max(-1)[0] * 3)
        att_1 = att_1 * rate_1.unsqueeze(-1) + _extra_att.transpose(1, 2)
        att = self.w_att(_att)
        rate = _extra_att.max(-1)[0] / (att.max(-1)[0] * 3)
        att = att * rate.unsqueeze(-1) + _extra_att
        _, indices = torch.sort(att, dim=-1, descending=True)
        _, indices_1 = torch.sort(att_1, dim=-1, descending=True)
        # 对于每个region对应的grid数量不一样,需要_mask
        indices = indices.masked_fill(_mask, -1) + 1
        indices = torch.cat([torch.zeros((b_s, r_2, 1), device=indices.device), indices], dim=-1).type(torch.long)
        indices_1 = indices_1.masked_fill(_mask_1, -1) + 1
        indices_1 = torch.cat([torch.zeros((b_s, g_2, 1), device=indices_1.device), indices_1], dim=-1).type(torch.long)
        _mask = torch.ones((b_s, r_2, g_2 + 1), device=indices.device) / 20.
        _mask = torch.scatter(_mask, -1, indices, torch.ones_like(_mask))[:, :, 1:]
        _mask_1 = torch.ones((b_s, g_2, r_2 + 1), device=indices_1.device) / 20.
        _mask_1 = torch.scatter(_mask_1, -1, indices_1, torch.ones_like(_mask_1))[:, :, 1:]
        # 如果这个region为0,那么对应的grid也应为0
        grid2region = self.layer_norm(self.feature_fusion(_region, _grid, extra_att, mask=_mask) + region)
        grid2region = F.layer_norm(self.multi_head_attention_2(grid2region, mask=sg_mask[1]) + grid2region, (region.shape[-1],))
        grid2region = F.layer_norm(self.feed_forward_2(grid2region) + grid2region, (region.shape[-1],))
        extra_att = extra_att.transpose(1, 2)
        region2grid = self.layer_norm_1(self.feature_fusion_1(_grid, _region, extra_att, mask=_mask_1) + grid)
        region2grid = F.layer_norm(self.multi_head_attention_1(region2grid) + region2grid, (grid.shape[-1],))
        region2grid = F.layer_norm(self.feed_forward_1(region2grid) + region2grid, (grid.shape[-1],))
        image = torch.cat([region2grid, grid2region], dim=1)
        sg_mask = torch.cat([torch.zeros(grid.shape[:2], dtype=torch.bool).to(grid.device), sg_mask[1]], dim=1)
        return image, sg_mask, sg, _sg_mask

class ParallelEncoder(nn.Module):
    def __init__(self, d_model:int, N_enc:int, word_emb, dropout, num_heads) -> None:
        super().__init__()
        self.word_emb = word_emb
        # self.grid_augmented_self_attention = GridAugmentedSelfAttention_new(num_heads)
        self.layers = nn.ModuleList([ParallelEncoderLayer(d_model, N_enc, self.word_emb, dropout, num_heads) for _ in range(1)])

    def init(self):
        for l in self.layers:
            l.init()
        # self.grid_augmented_self_attention.init()

    def forward(self, image, image_id=None, enti2attr=None, sub2obj2rela=None, sg_mask=None, regions=None, boxes=None):
        pos_att = None
        # pos_att = self.grid_augmented_self_attention(image)
        for l in self.layers:
            image, sg_mask, sg, _sg_mask = l(image, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes, pos_att)
        return image, sg_mask, sg, _sg_mask
