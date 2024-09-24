import spacy
import torch
import re
import sng_parser

from config.GlobalConfig import GlobalConfig
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, batch_size: int, d_model: int):
        super().__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        # self.nlp = spacy.load('en_core_web_sm')
        # self.regex = re.compile(r'([a-zA-Z]+)')
        self.parser = sng_parser.Parser('spacy', model='en_core_web_sm')
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.params = nn.Parameter(torch.randn((32, d_model)))
        self.softmax = nn.Softmax(dim=-1)
        self.max_x_len = 45
        self.linear_attr = nn.Linear(d_model * 2, d_model)
        self.linear_rela = nn.Linear(d_model * 3, d_model)
        self.linear_sub = nn.Linear(d_model * 3, d_model)
        self.linear_obj = nn.Linear(d_model * 3, d_model)

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def init(self):
        pass

    def forward(self, caption2vector, caption):
        x = []
        # for i in range(len(caption)):
        for i in range(self.batch_size):
            enti2attr, sub2rela2obj = self.get_scene_graph(caption[i])
            x_rela, x_enti = None, None
            _x = None
            x_attr = self.g_attr_fn(enti2attr, caption2vector[i, :, :], caption[i])
            if x_attr is not None:
                if len(sub2rela2obj) != 0:
                    x_rela = self.g_rela_fn(enti2attr, sub2rela2obj, caption2vector[i, :, :], caption[i])
                    x_enti = self.g_enti_fn(enti2attr, sub2rela2obj, caption2vector[i, :, :], caption[i])
                    _x = torch.cat([x_rela, x_attr, x_enti], dim=0).to(self.params.device)
                else:
                    _x = x_attr.to(self.params.device)
            else:
                _x = torch.zeros((1, self.d_model)).to(self.params.device)
            att = self.softmax(torch.matmul(_x, self.params.transpose(-1, -2)))
            _x = torch.matmul(att, self.params) 
            _len = self.max_x_len - _x.shape[0] 
            if _len > 0:
                _x = torch.cat([_x, torch.zeros((_len, self.d_model)).to(self.params.device)], dim=0)
            else:
                print(f"图信息x的长度<{_x.shape[0]}>超过了最大长度,已经截断！")
                # print(caption[i])
                _x = _x[:self.max_x_len, :]
            x.append(_x)
        return torch.stack(x, dim=0)

    def calc_one_enti(self, caption2vector, pos):
        t = torch.zeros((self.d_model)).to(caption2vector.device)
        for i in pos:
            if i >= caption2vector.shape[0]:
                continue
            t = t + caption2vector[i]
        t = t / len(pos)
        return t

    def get_pos(self, caption, word):
        caption = caption.split(" ")
        word = word.split(" ")
        pos = []
        for i, c in enumerate(caption):
            for j, w in enumerate(word):
                if w == c:
                    pos.append(i)
                    del word[j]
        if len(word) > 0:
            for i, c in enumerate(caption):
                for j, w in enumerate(word):
                    if w in c:
                        pos.append(i)
                        del word[j]
        if len(pos) == 0:
            print((caption,word))
        return pos

    def g_rela_fn(self, enti2attr, sub2rela2obj, caption2vector, caption):
        x_rela = []
        for obj, rela, sub in sub2rela2obj:
            obj_pos = self.get_pos(caption, list(enti2attr[int(obj)].keys())[0])
            sub_pos = self.get_pos(caption, list(enti2attr[int(sub)].keys())[0])
            rela_pos = self.get_pos(caption, rela)
            obj_vec = self.calc_one_enti(caption2vector, obj_pos)
            sub_vec = self.calc_one_enti(caption2vector, sub_pos)
            rela_vec = self.calc_one_enti(caption2vector, rela_pos)
            _x_rela = self.linear_rela(torch.cat([obj_vec, rela_vec, sub_vec], dim=-1))
            # _x_rela = obj_vec * 0.2 + rela_vec * 0.5 + sub_vec * 0.2
            x_rela.append(_x_rela)
        return torch.stack(x_rela, dim=0)

    def g_attr_fn(self, enti2attr, caption2vector, caption):
        x_attr = []
        for e2a in enti2attr:
            for enti, attr in e2a.items():
                enti_pos = self.get_pos(caption, enti)
                enti_vec = self.calc_one_enti(caption2vector, enti_pos)
                if len(attr) == 0:
                    x_attr.append(enti_vec)
                    continue
                attr_vec = sum([self.calc_one_enti(caption2vector, self.get_pos(caption, a)) for a in attr]) / len(attr)
                _x_attr = self.linear_attr(torch.cat([enti_vec, attr_vec], dim=-1))
                # _x_attr = enti_vec * 0.7 + attr_vec * 0.3
                x_attr.append(_x_attr)
        if len(x_attr) == 0:
            return None
        return torch.stack(x_attr, dim=0)

    def g_enti_fn(self, enti2attr, sub2rela2obj, caption2vector, caption):
        x_enti = []
        _x_enti = {}
        x_sub = self.g_sub_fn(enti2attr, sub2rela2obj, caption2vector, caption)
        x_obj = self.g_obj_fn(enti2attr, sub2rela2obj, caption2vector, caption)
        for i, (sub, rela, obj) in enumerate(sub2rela2obj):
            sub = list(enti2attr[int(sub)].keys())[0]
            if sub not in _x_enti:
                _x_enti[sub] = [x_sub[i, :]]
            else:
                _x_enti[sub].append(x_sub[i, :])
            obj = list(enti2attr[int(obj)].keys())[0]
            if obj not in _x_enti:
                _x_enti[obj] = [x_obj[i, :]]
            else:
                _x_enti[obj].append(x_obj[i, :])
        for k, v in _x_enti.items():
            x_enti.append(self.avg_pool(torch.stack(v, dim=0).unsqueeze(0).transpose(-1, -2)).transpose(-1, -2).squeeze(0))
        return torch.cat(x_enti, dim=0)

    def g_sub_fn(self, enti2attr, sub2rela2obj, caption2vector, caption):
        x_sub = []
        for sub, rela, obj in sub2rela2obj:
            sub_pos = self.get_pos(caption, list(enti2attr[int(sub)].keys())[0])
            rela_pos = self.get_pos(caption, rela)
            obj_pos = self.get_pos(caption, list(enti2attr[int(obj)].keys())[0])
            obj_vec = self.calc_one_enti(caption2vector, obj_pos)
            sub_vec = self.calc_one_enti(caption2vector, sub_pos)
            rela_vec = self.calc_one_enti(caption2vector, rela_pos)
            _x_sub = self.linear_sub(torch.cat([obj_vec, rela_vec, sub_vec], dim=-1))
            # _x_sub = obj_vec * 0.2 + rela_vec * 0.3 + sub_vec * 0.5
            x_sub.append(_x_sub)
        return torch.stack(x_sub, dim=0)

    def g_obj_fn(self, enti2attr, sub2rela2obj, caption2vector, caption):
        x_obj = []
        for sub, rela, obj in sub2rela2obj:
            sub_pos = self.get_pos(caption, list(enti2attr[int(sub)].keys())[0])
            rela_pos = self.get_pos(caption, rela)
            obj_pos = self.get_pos(caption, list(enti2attr[int(obj)].keys())[0])
            obj_vec = self.calc_one_enti(caption2vector, obj_pos)
            sub_vec = self.calc_one_enti(caption2vector, sub_pos)
            rela_vec = self.calc_one_enti(caption2vector, rela_pos)
            _x_obj = self.linear_obj(torch.cat([obj_vec, rela_vec, sub_vec], dim=-1))
            # _x_obj = obj_vec * 0.5 + rela_vec * 0.3 + sub_vec * 0.2
            x_obj.append(_x_obj)
        return torch.stack(x_obj, dim=0)

    def get_scene_graph(self, caption:str):
        graph = self.parser.parse(caption)
        # sng_parser.tprint(graph)
        enti2attr = []
        sub2rela2obj = []
        for ent in graph["entities"]:
            attr = [e["span"] for e in ent["modifiers"]]
            enti2attr.append({ent["head"]: attr})
        for rela in graph["relations"]:
            cell = [rela["subject"], rela["relation"], rela["object"]]
            sub2rela2obj.append(cell)
        return enti2attr, sub2rela2obj
    
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