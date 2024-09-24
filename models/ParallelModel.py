import torch

from torch import nn
from .ParallelEncoder_1 import ParallelEncoder
from .utils import get_activation_function
from config.GlobalConfig import GlobalConfig
from .enums import DataType
from models.utils import CalcTime
from .ParallelDecoder_fusion import ParallelDecoder
from torchvision import models

class ParallelModel(nn.Module):
    def __init__(self, N_enc:int, N_dec:int, num_heads:int, is_use_hdf5:bool) -> None:
        super().__init__()
        self.d_model = GlobalConfig.d_model
        self.is_use_hdf5 = is_use_hdf5
        if not is_use_hdf5:
            self.resnet = models.resnet152(pretrained=True)
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
            self.resnet.add_module('avgpool', nn.AdaptiveAvgPool2d((12, 12)))
            self.resnet_ext = nn.Linear(2048, GlobalConfig.d_model)
        else:
            # self.grid_project = nn.Sequential(
            #     nn.Linear(1536, GlobalConfig.d_model),
            #     get_activation_function("relu"),
            #     nn.Dropout(p=GlobalConfig.dropout),
            #     nn.LayerNorm(GlobalConfig.d_model)
            # )
            self.grid_project = nn.Sequential(
                nn.Linear(2048, GlobalConfig.d_model),
                get_activation_function("relu"),
                nn.Dropout(p=GlobalConfig.dropout),
                nn.LayerNorm(GlobalConfig.d_model)
            )
            self.region_project = nn.Sequential(
                nn.Linear(2048, GlobalConfig.d_model),
                get_activation_function("relu"),
                nn.Dropout(p=GlobalConfig.dropout),
                nn.LayerNorm(GlobalConfig.d_model)
            )
        self.word_emb = nn.Sequential(
            nn.Embedding(GlobalConfig.vocab_size, GlobalConfig.d_model,padding_idx=GlobalConfig.padding_idx),
            get_activation_function(),
            # nn.Dropout(GlobalConfig.dropout)
        )
        # self.encoder = ParallelEncoder(batch_size, GlobalConfig.d_model, N_enc, self.word_emb, GlobalConfig.dropout, num_heads)
        self.encoder = ParallelEncoder(GlobalConfig.d_model, N_enc, self.word_emb, GlobalConfig.dropout, num_heads)
        self.decoder = ParallelDecoder(GlobalConfig.d_model, N_dec, self.word_emb, GlobalConfig.dropout, num_heads)
        # self.adaptive_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.init()

    def init(self):
        if self.is_use_hdf5:
            nn.init.xavier_uniform_(self.grid_project[0].weight)
            nn.init.xavier_uniform_(self.region_project[0].weight)

            nn.init.constant_(self.grid_project[0].bias, 0)
            nn.init.constant_(self.region_project[0].bias, 0)
        
        self.encoder.init()
        self.decoder.init()

    def forward(self, enc_input, caption2i=None,  enti2attr=None, sub2obj2rela=None, image_id=None, sg_mask=None, regions=None ,boxes=None):
        #
        if self.is_use_hdf5:
            enc_input = self.grid_project(enc_input)
            regions = self.region_project(regions)
        else:
            enc_input = self.resnet(enc_input).reshape(self.batch_size, 144, -1)
            enc_input = self.resnet_ext(enc_input)
        enc_output, sg_mask, sg, _sg_mask = self.encoder(enc_input, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes)
        #
        if sg_mask is not None:
            enc_output = enc_output.masked_fill(sg_mask.unsqueeze(-1), 0.)
            enc_output_mean = enc_output.sum(dim=1) / (~sg_mask).sum(dim=1).unsqueeze(-1)
        else:
            enc_output_mean = enc_output.mean(dim=1)
        if GlobalConfig.mode == DataType.TRAIN:
            caption2vector = self.word_emb(caption2i)
            dec_output, out_caption2i = self.decoder(enc_output_mean, enc_output, caption2i, caption2vector, sg_mask, sg, _sg_mask)
        elif GlobalConfig.mode in { DataType.TEST, DataType.VAL }:
            dec_output, out_caption2i = self.decoder.sample(enc_output_mean, enc_output, sg_mask=sg_mask, sg=sg, _sg_mask=_sg_mask)
        return dec_output, out_caption2i

    def step(self, enc_input, caption2i=None,  enti2attr=None, sub2obj2rela=None, image_id=None):
        caption2vector = None
        if GlobalConfig.mode == DataType.TRAIN:
            caption2vector = self.word_emb(caption2i)
        #
        enc_input = self.grid_project(enc_input)
        # enc_input = self.feature_project(enc_input.transpose(-1, -2).unsqueeze(-1)).squeeze(-1).transpose(-1, -2)
        enc_output = self.encoder(enc_input, image_id, enti2attr, sub2obj2rela)
        #
        enc_output_mean =self.adaptive_avg_pooling(enc_output.transpose(-1, -2)).transpose(-1, -2)
        dec_output = self.decoder.step(enc_output_mean, enc_output, caption2i, caption2vector)
        return dec_output
    