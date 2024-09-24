import spacy
import torch
import re
import sng_parser
import os
import numpy as np
import torch.nn as nn
import time
import torch.nn.functional as F

from config.GlobalConfig import GlobalConfig
from typing import List
from data.vocab.Vocab import Vocab
from models.enums import DataType
from models.utils import CalcTime
from models.utils import get_activation_function

class GCN(nn.Module):
    def __init__(self, d_model: int, word_emb):
        super().__init__()
        self.word_emb = word_emb
        self.d_model = d_model
        # self.nlp = spacy.load('en_core_web_sm')
        # self.regex = re.compile(r'([a-zA-Z]+)')
        # self.parser = sng_parser.Parser('spacy', model='en_core_web_sm')
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.params = nn.Parameter(torch.randn((1000, d_model)))
        # self.max_x_len = 144
        # no attr:99 rela:90 enti:55
        # enti:63
        # have attr:34 rela:56 enti:27
        self.enti_max_x_len = 63
        rela_len = 472
        self.rela_embed = nn.Sequential(
            nn.Embedding(rela_len, d_model),
            get_activation_function(),
            # nn.Dropout(GlobalConfig.dropout)
        )
        self.linear_attr = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            get_activation_function(),
            nn.Dropout(GlobalConfig.dropout)
        )
        self.linear_rela = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            get_activation_function(),
            nn.Dropout(GlobalConfig.dropout)
        )
        self.linear_sub = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            get_activation_function(),
            nn.Dropout(GlobalConfig.dropout)
        )
        self.linear_obj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            get_activation_function(),
            nn.Dropout(GlobalConfig.dropout)
        )
        self.ssg_obj_obj = nn.Sequential(
            nn.Linear(d_model, d_model),
            get_activation_function(),
            nn.Dropout(GlobalConfig.dropout)
        )
        self.vocab = Vocab()
        # sg_dict_path = "data/features/coco_pred_sg_rela.npy"
        # sg_dict = np.load(sg_dict_path, allow_pickle=True, encoding="latin1")[()]
        # self.sg_i2w = sg_dict["i2w"]
        # self.sg_rela_dict = sg_dict["rela_dict"]
        # self.sg_w2i = sg_dict["w2i"]

    def init(self):
        nn.init.normal_(self.params, 0, 1)
        # nn.init.xavier_uniform_(self.params)
        nn.init.xavier_uniform_(self.linear_rela[0].weight)
        nn.init.xavier_uniform_(self.linear_sub[0].weight)
        nn.init.xavier_uniform_(self.linear_obj[0].weight)
        nn.init.xavier_uniform_(self.linear_attr[0].weight)
        nn.init.xavier_uniform_(self.ssg_obj_obj[0].weight)

        nn.init.constant_(self.linear_attr[0].bias, 0)
        nn.init.constant_(self.linear_rela[0].bias, 0)
        nn.init.constant_(self.linear_sub[0].bias, 0)
        nn.init.constant_(self.linear_obj[0].bias, 0)
        nn.init.constant_(self.ssg_obj_obj[0].bias, 0)
        
    def forward(self, _enti2attr, _sub2obj2rela, sg_mask=None):
        enti2attr, sub2obj2rela = self.sg2vector(_enti2attr, _sub2obj2rela)
        # obj_obj = self.sg2entivector(enti2attr)
        x_attr = self.g_attr_fn(enti2attr)
        x_rela = self.g_rela_fn(sub2obj2rela)
        x_enti = self.g_enti_fn(sub2obj2rela, _sub2obj2rela, enti2attr[:, :, :self.d_model].clone(), _enti2attr[:, :, 0].squeeze(-1))
        # update
        # enti2attr, sub2obj2rela = self.update_graph(enti2attr.clone(), sub2obj2rela.clone(), _sub2obj2rela, x_attr, x_rela, x_enti, sg_mask[0])
        sg_mask = torch.cat([sg_mask[0], sg_mask[1], sg_mask[1]], dim=1)
        x = torch.cat([x_rela, x_attr, x_enti], dim=1)
        att = F.softmax(torch.matmul(x, self.params.transpose(-1, -2)), -1)
        _x = F.relu(F.dropout(torch.matmul(att, self.params), GlobalConfig.dropout)) + x
        x = _x.masked_fill(sg_mask.unsqueeze(-1).repeat(1, 1, self.d_model), 0.)
        return x, sg_mask

    def update_graph(self, enti2attr, sub2obj2rela, _sub2obj2rela, x_attr, x_rela, x_enti, x_rela_mask):
        _rela_len = x_rela_mask.shape[1]
        enti2attr[:, :, self.d_model:] = x_attr
        enti2attr[:, :, self.d_model:] = x_enti
        sub2obj2rela[:, :, self.d_model*2:] = x_rela
        for j, (s2r2o) in enumerate(_sub2obj2rela):
            if GlobalConfig.mode == DataType.TRAIN:
                if j % 5 == 0:
                    for i, (sub, obj, rela) in enumerate(s2r2o):
                        if x_rela_mask[j, i]:
                            break
                        sub, obj = int(sub.item()), int(obj.item())
                        if sub >= _rela_len or obj >= _rela_len:
                            break
                        sub2obj2rela[j, i, :self.d_model] = x_enti[j, sub]
                        sub2obj2rela[j, i, self.d_model:self.d_model*2] = x_enti[j, obj]
            elif GlobalConfig.mode in { DataType.VAL, DataType.TEST }:
                for i, (sub, obj, rela) in enumerate(s2r2o):
                    if x_rela_mask[j, i]:
                        break
                    sub, obj = int(sub.item()), int(obj.item())
                    if sub >= _rela_len or obj >= _rela_len:
                            break
                    sub2obj2rela[j, i, :self.d_model] = x_enti[j, sub]
                    sub2obj2rela[j, i, self.d_model:self.d_model*2] = x_enti[j, obj]
        return enti2attr, sub2obj2rela

    def get_vector(self, word, is_rela=False):
        vector = None
        if is_rela:
            vector = self.rela_embed(word.int())
        else:
            vector = self.vocab.vocab2i_fn(word)
            vector = torch.tensor(vector, dtype=torch.int32).to(self.word_emb[0].weight.device)
            vector = self.word_emb(vector)
        return vector   

    def sg2entivector(self, enti2attr):
        _obj_obj = enti2attr[:, :, :self.d_model]
        _obj_obj = self.ssg_obj_obj(_obj_obj)
        return _obj_obj

    def sg2vector(self, enti2attr, sub2obj2rela):
        bs_seq = enti2attr.shape[:2]
        _enti2attr = self.word_emb(enti2attr.long()).reshape(*bs_seq, -1)
        #
        bs_seq = sub2obj2rela.shape[:2]
        _sub2obj = self.word_emb(sub2obj2rela.long()[:, :, :2])
        _rela = self.rela_embed(sub2obj2rela.long()[:, :, 2])
        _sub2obj2rela = torch.cat([_sub2obj, _rela.unsqueeze(-2)], dim=-2).reshape(*bs_seq, -1)
        return _enti2attr, _sub2obj2rela

    def g_rela_fn(self, sub2obj2rela):
        x_rela = (sub2obj2rela[:, :, self.d_model:self.d_model*2] + self.linear_rela(sub2obj2rela)) / 2
        return x_rela

    def g_attr_fn(self, enti2attr):
        x_attr = self.linear_attr(enti2attr)
        return x_attr

    def g_enti_fn(self, sub2obj2rela, _sub2obj2rela, obj_vector, obj_index):
        b_s, seq_len = obj_index.shape[:2]
        x_sub = self.g_sub_fn(sub2obj2rela)
        x_obj = self.g_obj_fn(sub2obj2rela)
        # 关系中的主体与客体对应的向量组成一个list
        x_enti = torch.cat([x_sub, x_obj], dim=-2)
        # 关系中的主体与客体组成一个list
        pos = _sub2obj2rela[:, :, :2].transpose(-1, -2).reshape(b_s, -1)
        for i in range(seq_len):
            # 将单个实体对应的主体客体时的向量进行mask
            mask = (pos != obj_index[:, i].unsqueeze(-1))
            mask_sum = torch.sum(~mask, dim=-1).unsqueeze(-1)
            vector_sum = torch.sum(x_enti.masked_fill(mask.unsqueeze(-1), 0.), dim=-2).squeeze(-2)
            obj_vector[:, i, :] = (obj_vector[:, i, :] + vector_sum) / (mask_sum + 1.)
        return obj_vector

    def g_enti_fn_1(self, sub2obj2rela, _sub2obj2rela, x_rela_mask, obj_obj, obj_index):
        x_sub = self.g_sub_fn(sub2obj2rela)
        x_obj = self.g_obj_fn(sub2obj2rela)
        for j, (s2r2o) in enumerate(_sub2obj2rela):
            obj_num = torch.ones((obj_obj.shape[1]), dtype=torch.float32).to(sub2obj2rela.device)
            if GlobalConfig.mode == DataType.TRAIN:
                if j % 5 == 0:
                    index = obj_index[j, :].tolist()
                    for i, (sub, obj, rela) in enumerate(s2r2o):
                        if x_rela_mask[j, i]:
                            break
                        try:
                            sub, obj = index.index(sub.item()), index.index(obj.item())
                        except ValueError:
                            continue
                        obj_obj[j, sub] = obj_obj[j, sub, :] + x_sub[j, i, :]
                        obj_num[sub] = obj_num[sub] + 1.
                        obj_obj[j, obj] = obj_obj[j, obj, :] + x_obj[j, i, :]
                        obj_num[obj] = obj_num[obj] + 1.
                    obj_obj[j, :] = obj_obj[j, :] / obj_num.unsqueeze(-1)
                else:
                    obj_obj[j, :] = obj_obj[j - 1, :]
            elif GlobalConfig.mode in { DataType.VAL, DataType.TEST }:
                index = obj_index[j, :].tolist()
                for i, (sub, obj, rela) in enumerate(s2r2o):
                    if x_rela_mask[j, i]:
                        break
                    try:
                        sub, obj = index.index(sub.item()), index.index(obj.item())
                    except ValueError:
                        continue
                    obj_obj[j, sub] = obj_obj[j, sub, :] + x_sub[j, i, :]
                    obj_num[sub] = obj_num[sub] + 1.
                    obj_obj[j, obj] = obj_obj[j, obj, :] + x_obj[j, i, :]
                    obj_num[obj] = obj_num[obj] + 1.
                obj_obj[j, :] = obj_obj[j, :] / obj_num.unsqueeze(-1)
        return obj_obj

    def g_sub_fn(self, sub2obj2rela):
        x_sub = self.linear_sub(sub2obj2rela)
        return x_sub

    def g_obj_fn(self, sub2obj2rela):
        x_obj = self.linear_obj(sub2obj2rela)
        return x_obj

    def get_scene_graph(self, image_id):
        # 
        sg_root = "data/features/coco_pred_sg"
        sg_dict_path = "data/features/coco_pred_sg_rela.npy"

        enti2attr, sub2obj2rela = {}, {}
        for k, (_id) in enumerate(image_id):
            # sg
            sg = np.load(os.path.join(sg_root, f"{_id}.npy"), allow_pickle=True, encoding="latin1")[()]
            sg_rela = sg['rela_matrix']
            sg_attr = sg['obj_attr']

            # dict
            sg_dict = np.load(sg_dict_path, allow_pickle=True, encoding="latin1")[()]
            sg_i2w = sg_dict["i2w"]
            sg_rela_dict = sg_dict["rela_dict"]
            sg_w2i = sg_dict["w2i"]

            _enti2attr, _sub2obj2rela = {}, []

            sg_obj = []
            for i, obj, attr in sg_attr:
                obj = sg_i2w[obj]
                sg_obj.append(obj)
                attr = sg_i2w[attr]
                _enti2attr[obj] = [attr]
            enti2attr[_id + str(k)] = _enti2attr

            for i, j, rela in sg_rela:
                sub = sg_obj[int(i)]
                rela = sg_rela_dict[rela]
                obj = sg_obj[int(j)]
                _sub2obj2rela.append([sub, rela, obj])
            sub2obj2rela[_id + str(k)] = _sub2obj2rela

        return enti2attr, sub2obj2rela
    
    ''' no use
    所有关系的集合
    '''
    def get_edge_index(self, caption):
        caption = list(caption)
        batch_size = len(caption)
        _max:int = 0
        pos = []
        for i in range(batch_size):
            # 将长度超过max_seq_len-2的caption截断
            caption[i] = " ".join(caption[i].split(" ")[:GlobalConfig.max_seq_len - 2])
            pos.append(self.regex.findall(caption[i]))
            _len = len(pos[i])
            _max = _len if _len > _max else _max
        edge_index = torch.ones((batch_size, 2, _max + 1), dtype=torch.long) * -1
        for i in range(batch_size):
            edge_index[i, :, -1] = torch.tensor([0,0])
            doc = self.nlp(caption[i])
            relations = [((r.head.i,r.head.text), (r.i, r.text), r.dep_) for r in doc]
            for j, (_object, subject, relation) in enumerate(relations):
                if j >= _max or (_object[1] not in pos[i]) or (subject[1] not in pos[i]):
                    break
                edge_index[i, :, j] = torch.tensor([_object[0], subject[0]])
                edge_index[i, :, -1] = edge_index[i, :, -1] + torch.tensor([1,1])
        return edge_index