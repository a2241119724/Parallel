#引用正则表达式模块
import re
import torch
import os
import requests
import time
import json

from torch import nn
from config.GlobalConfig import GlobalConfig
from pprint import pprint
from data.evaluation.tokenizer import PTBTokenizer

def get_activation_function(name=""):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "softmax":
        return nn.Softmax(-1)
    elif name == "log_softmax":
        return nn.LogSoftmax(-1)
    elif name== "elu":
        return nn.ELU(inplace=True)
    else:
        return nn.Identity()

def print_parameter_count(model, is_simplify=False, is_print_all=False, is_print_detail=False,contain_str=None):
    regex = re.compile("(\\.)")
    params = model.named_parameters()
    select_parameters_count = 0
    if contain_str != None:
        params = filter(lambda it: contain_str in it[0], params)
    if is_simplify:
        print("total_parameters_count: %d" % sum(p.numel() for p in model.parameters()))
        print("train_parameters_count: %d" % sum(p.numel() for p in model.parameters() if p.requires_grad))
        if is_print_all:
            for name, param in params:
                select_parameters_count = select_parameters_count + param.numel()
                if is_print_detail:
                    print(name + "\t\t\t\t\t\t\t\t" + str(param.numel()))
                    continue
                if len(list(regex.finditer(name))) > 1:
                    index = list(regex.finditer(name))[-2].start()
                else:
                    index = 0
                print(name[index:] + "\t\t\t\t\t\t\t\t" + str(param.numel()))
            if contain_str != None:
                print("select_parameters_count: %d" % select_parameters_count)
    else:
        count = {}
        count["total_parameters_count"] = sum(p.numel() for p in model.parameters())
        count["train_parameters_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if is_print_all:
            for name, param in params:
                select_parameters_count = select_parameters_count + param.numel()
                if is_print_detail:
                    count[name] = param.numel()
                    continue
                if len(list(regex.finditer(name))) > 1:
                    index = list(regex.finditer(name))[-2].start()
                else:
                    index = 0
                count[name[index:]] = param.numel()
            if contain_str != None:
                print("select_parameters_count: %d" % select_parameters_count)
        pprint(count)

def _position_embedding(input, d_model):
    device = input.device
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

# sin cos绝对位置编码
def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = _position_embedding(pos, d_model)

    if padding_idx is not None:
        out[padding_idx] = 0
    return out

def calc_code_lines(path:str="./"):
    total = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py'): # 根据不同的编程语言修改此处条件
                with open(os.path.join(root, file), 'r') as f:
                    lines = len(f.readlines())
                    total += lines
    print("项目代码行数：", total)

def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def pre_handle_caption():
    # caption = caption.replace(",", "").replace(".","")\
    #     .replace("?","").replace("!","").replace("'s"," 's")\
    #     .replace('"',"").replace("\n","").replace("'","")\
    #     .replace("-"," ").replace("/"," ").replace(";","")\
    #     .replace("(","").replace(")","")\
    #     .lower()
    ptb = PTBTokenizer()
    regex = re.compile(r"([1-9]\d*?).jpg$")
    train_anno = json.load(open("./data/annotations/coco_karpathy_train2014.json", "r", encoding="utf-8"))
    val_anno = json.load(open("./data/annotations/coco_karpathy_val2014.json", "r", encoding="utf-8"))
    test_anno = json.load(open("./data/annotations/coco_karpathy_test2014.json", "r", encoding="utf-8"))
    #
    dict_train = {}
    for anno in train_anno:
        dict_train[anno["image_id"]] = [anno["caption"]]
    dict_train = ptb.tokenize(dict_train)
    _train_anno = []
    for anno in train_anno:
        caption = dict_train[anno["image_id"]]
        _train_anno.append({
            "caption": caption[0],
            "image": anno["image"],
            "image_id": anno["image_id"].split("_")[1]
        })
    json.dump(_train_anno,open("./data/annotations/coco_lab_karpathy_train2014.json", "w", encoding="utf-8"))
    #
    dict_val = {}
    for anno in val_anno:
        image_id = regex.findall(anno['image'])[-1]
        dict_val[image_id] = anno["caption"]
    dict_val = ptb.tokenize(dict_val)
    _val_anno = []
    for anno in val_anno:
        image_id = regex.findall(anno['image'])[-1]
        caption = dict_val[image_id]
        _val_anno.append({
            "caption": caption,
            "image": anno["image"],
            "image_id": image_id
        })
    json.dump(_val_anno,open("./data/annotations/coco_lab_karpathy_val2014.json", "w", encoding="utf-8"))
    #
    dict_test = {} 
    for anno in test_anno:
        image_id = regex.findall(anno['image'])[-1]
        dict_test[image_id] = anno["caption"]
    dict_test = ptb.tokenize(dict_test)
    _test_anno = []
    for anno in test_anno:
        image_id = regex.findall(anno['image'])[-1]
        caption = dict_test[image_id]
        _test_anno.append({
            "caption": caption,
            "image": anno["image"],
            "image_id": image_id
        })
    json.dump(_test_anno,open("./data/annotations/coco_lab_karpathy_test2014.json", "w", encoding="utf-8"))

class CalcTime():
    step = 0
    total_time = 0.
    
    def __init__(self, info="" , reset=False) -> None:
        self.info = info
        if reset:
            CalcTime.step = 0
            CalcTime.total_time = 0.

    def __enter__(self):
        CalcTime.step = CalcTime.step + 1
        self.start = time.time()
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        print("----------- " + str(CalcTime.step) + " -----------")
        curr_time = time.time() - self.start
        print(f"{self.info} curr time cost: \t", curr_time)
        CalcTime.total_time = CalcTime.total_time + curr_time
        print(f"{self.info} avg time cost: \t", CalcTime.total_time / CalcTime.step)

def get_grids_pos(grid_size=12, device="cuda"):
    x = torch.arange(0, grid_size).float().to(device)
    y = torch.arange(0, grid_size).float().to(device)
    px_min = x.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    px_max = px_min + 1
    py_max = py_min + 1

    x_min = px_min.view(1, -1, 1) / grid_size
    y_min = py_min.view(1, -1, 1) / grid_size
    x_max = px_max.view(1, -1, 1) / grid_size
    y_max = py_max.view(1, -1, 1) / grid_size
    return x_min, y_min, x_max, y_max
