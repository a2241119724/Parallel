import os
import torch
import json
import torch.nn.functional as F
import re
import h5py
import time
import numpy as np
import csv
import base64

from typing import Tuple
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config.GlobalConfig import GlobalConfig
from models.enums import DataType
from data.vocab.Vocab import Vocab
# from models.utils import pre_handle_caption

class MsCocoDataset(Dataset):
    '''
    * @param image_path: 图片路径前缀
    * @param annotation_path: 注解路径前缀
    * @param data_type: 数据类型
    '''
    def __init__(self, image_path:str, annotation_path:str=None, data_type: DataType=DataType.TRAIN, 
            device=None, hdf5_grid:str=None, is_use_hdf5=False):
        self.is_region = True
        self.is_sg = True
        self.is_grid = True
        self.device = device
        # if not os.path.exists(image_path):
        #     raise Exception("Image path not exists!")
        if annotation_path is not None and not os.path.exists(annotation_path):
            raise Exception("Annotation path not exists!")
        self.data_type:DataType = data_type
        #
        annotation_path = annotation_path if annotation_path is not None else image_path
        self.image_path = os.path.join(image_path)
        annotation_path = os.path.join(annotation_path, 'coco_lab_karpathy_' + data_type.value + '2014.json')
        with open(annotation_path) as f:
            self.image_annotation = json.load(f)
            self.len = len(self.image_annotation)
        ## vocab
        # self.one_hot = F.one_hot
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab = Vocab()
        GlobalConfig.vocab_size =  self.vocab.vocab_size
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # self.regex = re.compile(r"([1-9]\d*?).jpg$")
        self.is_use_hdf5 = is_use_hdf5
        if self.is_grid and is_use_hdf5:
            self.hdf5_grid = h5py.File(hdf5_grid, 'r')
        if self.is_sg:
            # self.hdf5_sg = h5py.File("data/features/sg_no.h5", 'r')
            self.hdf5_sg = h5py.File("data/features/sg_str.h5", 'r')
            #
            # sg_dict_path = "data/features/coco_pred_sg_rela.npy"
            # sg_dict = np.load(sg_dict_path, allow_pickle=True, encoding="latin1")[()]
            # self.sg_i2w = sg_dict["i2w"]
            # self.sg_rela_dict = sg_dict["rela_dict"]
            # self.sg_w2i = sg_dict["w2i"]
        #
        self.loop = 5
        self.image = torch.zeros((1, 1), dtype=torch.float32)
        self.image_id = None
        self.enti2attr = torch.zeros((1, 2), dtype=torch.int32)
        self.sub2obj2rela = torch.zeros((1, 3), dtype=torch.int32)
        self.regions = torch.zeros((1, 1), dtype=torch.float32)
        self.boxes = torch.zeros((1, 1), dtype=torch.float32)
        # self.attr_max_x_len = 99
        # self.rela_max_x_len = 90
        self.attr_max_x_len = 30
        self.rela_max_x_len = 20
        # self.attr_max_count = 10
        #
        if self.is_region:
            # hdf5_region = os.path.join("./data/features/trainval", 'karpathy_resnet101_faster_rcnn_genome.hdf5')
            # self.hdf5_region = h5py.File(hdf5_region, 'r')
            self.hdf5_region = self.hdf5_grid

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        caption = self.image_annotation[index]['caption']
        image, image_id = None, None
        enti2attr, sub2obj2rela = [], []
        # _sub2obj2rela = []

        if self.data_type == DataType.TRAIN:
            if self.loop in [0, 5]:
                self.loop = 5
                image_id = self.image_annotation[index]['image_id']
                if self.is_grid:
                    if not self.is_use_hdf5:
                        image_path = os.path.join(self.image_path, self.image_annotation[index]['image'])
                        if not os.path.exists(image_path):
                            raise Exception("Image path not exists!")
                        image = Image.open(image_path, mode='r').convert('RGB')
                        image = self.transform(image)
                    else:
                        # image = torch.from_numpy(self.hdf5_grid[f"{image_id}_features"][()]).to(self.device)
                        image = torch.from_numpy(self.hdf5_grid[f"{image_id}_grids"][()]).to(self.device)
                    self.image = image

                if self.is_sg:
                    sub2obj2rela = torch.tensor(self.hdf5_sg[f"{image_id}_rela_matrix"][()], dtype=torch.long).to(self.device)
                    enti2attr = torch.tensor(self.hdf5_sg[f"{image_id}_obj_attr"][()], dtype=torch.long).to(self.device)
                    _len = self.attr_max_x_len - len(enti2attr)
                    if _len > 0:
                        # enti2attr = torch.cat([enti2attr, torch.zeros((_len, enti2attr.shape[-1]), dtype=torch.int32)], dim=0)
                        enti2attr = torch.cat([enti2attr, torch.zeros((_len, 2), dtype=torch.int32).to(self.device)], dim=0)
                    elif _len < 0:
                        enti2attr = enti2attr[:self.attr_max_x_len]
                    #
                    _len = self.rela_max_x_len - len(sub2obj2rela)
                    if _len > 0:
                        # sub2obj2rela = torch.cat([sub2obj2rela, torch.zeros((_len, sub2obj2rela.shape[-1]), dtype=torch.int32)], dim=0)
                        sub2obj2rela = torch.cat([sub2obj2rela, torch.zeros((_len, 3), dtype=torch.int32).to(self.device)], dim=0)
                    elif _len < 0:
                        sub2obj2rela = sub2obj2rela[:self.rela_max_x_len]
                    self.enti2attr = enti2attr
                    self.sub2obj2rela = sub2obj2rela

                if self.is_region:
                    regions = torch.tensor(self.hdf5_region[f"{image_id}_features"][()], dtype=torch.float32).to(self.device)
                    boxes = torch.tensor(self.hdf5_region[f"{image_id}_boxes"][()], dtype=torch.float32).to(self.device)
                    #
                    _len = self.attr_max_x_len - len(regions)
                    if _len > 0:
                        regions = torch.cat([regions, torch.zeros((_len, regions.shape[-1]), dtype=torch.float32).to(self.device)], dim=0)
                    elif _len < 0:
                        regions = regions[:self.attr_max_x_len]
                    #
                    _len = self.attr_max_x_len - len(boxes)
                    if _len > 0:
                        boxes = torch.cat([boxes, torch.zeros((_len, boxes.shape[-1]), dtype=torch.float32).to(self.device)], dim=0)
                    elif _len < 0:
                        boxes = boxes[:self.attr_max_x_len]
                    self.regions = regions
                    self.boxes = boxes
                
                self.image_id = image_id

            self.loop = self.loop - 1
            caption2i = self.vocab.caption2i(caption)
            return self.image, self.add_pad(torch.tensor(caption2i, dtype=torch.long)), self.image_id, caption, self.enti2attr, self.sub2obj2rela, self.regions, self.boxes
        elif self.data_type in  { DataType.TEST, DataType.VAL }:
            image_id = self.image_annotation[index]['image_id']
            if self.is_grid:
                if not self.is_use_hdf5:
                    image_path = os.path.join(self.image_path, self.image_annotation[index]['image'])
                    if not os.path.exists(image_path):
                        raise Exception("Image path not exists!")
                    image = Image.open(image_path, mode='r').convert('RGB')
                    image = self.transform(image)
                else:
                    image = torch.from_numpy(self.hdf5_grid[f"{image_id}_grids"][()]).to(self.device)
                self.image = image

            if self.is_sg:
                sub2obj2rela = torch.tensor(self.hdf5_sg[f"{image_id}_rela_matrix"][()], dtype=torch.long).to(self.device)
                enti2attr = torch.tensor(self.hdf5_sg[f"{image_id}_obj_attr"][()], dtype=torch.long).to(self.device)
                _len = self.attr_max_x_len - len(enti2attr)
                if _len > 0:
                    # enti2attr = torch.cat([enti2attr, torch.zeros((_len, enti2attr.shape[-1]), dtype=torch.int32)], dim=0)
                    enti2attr = torch.cat([enti2attr, torch.zeros((_len, 2), dtype=torch.int32).to(self.device)], dim=0)
                elif _len < 0:
                    enti2attr = enti2attr[:self.attr_max_x_len]
                #
                _len = self.rela_max_x_len - len(sub2obj2rela)
                if _len > 0:
                    # sub2obj2rela = torch.cat([sub2obj2rela, torch.zeros((_len, sub2obj2rela.shape[-1]), dtype=torch.int32)], dim=0)
                    sub2obj2rela = torch.cat([sub2obj2rela, torch.zeros((_len, 3), dtype=torch.int32).to(self.device)], dim=0)
                elif _len < 0:
                    sub2obj2rela = sub2obj2rela[:self.rela_max_x_len]
                self.enti2attr = enti2attr
                self.sub2obj2rela = sub2obj2rela

            if self.is_region:
                regions = torch.tensor(self.hdf5_region[f"{image_id}_features"][()], dtype=torch.float32).to(self.device)
                boxes = torch.tensor(self.hdf5_region[f"{image_id}_boxes"][()], dtype=torch.float32).to(self.device)
                #
                _len = self.attr_max_x_len - len(regions)
                if _len > 0:
                    regions = torch.cat([regions, torch.zeros((_len, regions.shape[-1]), dtype=torch.float32).to(self.device)], dim=0)
                elif _len < 0:
                    regions = regions[:self.attr_max_x_len]
                #
                _len = self.attr_max_x_len - len(boxes)
                if _len > 0:
                    boxes = torch.cat([boxes, torch.zeros((_len, boxes.shape[-1]), dtype=torch.float32).to(self.device)], dim=0)
                elif _len < 0:
                    boxes = boxes[:self.attr_max_x_len]
                self.regions = regions
                self.boxes = boxes
            return self.image, caption, image_id, self.enti2attr, self.sub2obj2rela, self.regions, self.boxes
        
    def collate_val_test_fn(self):
        def collate(batch):
            batch_size = len(batch)
            image, caption, image_id, enti2attr, sub2obj2rela, regions, boxes = zip(*batch)
            image = torch.stack(image, dim=0)
            enti2attr = torch.stack(enti2attr, dim=0)
            sub2obj2rela = torch.stack(sub2obj2rela, dim=0)
            regions = torch.stack(regions, dim=0)
            boxes = torch.stack(boxes, dim=0)
            x_attr_mask = torch.zeros((batch_size, self.attr_max_x_len), dtype=torch.bool)
            x_rela_mask = torch.zeros((batch_size, self.rela_max_x_len), dtype=torch.bool)
            if self.is_sg:
                x_attr_mask = ~torch.sum(enti2attr, dim=-1).bool()
                x_rela_mask = ~torch.sum(sub2obj2rela, dim=-1).bool()

            if self.is_region:
                x_attr_mask = ~torch.sum(regions, dim=-1).bool()
            return image, caption, image_id, enti2attr, sub2obj2rela, [x_rela_mask, x_attr_mask], regions, boxes
        return collate
    
    def collate_sg_train_fn(self):
        def collate(batch):
            batch_size = len(batch)
            image, caption2i, image_id, caption, enti2attr, sub2obj2rela, regions, boxes = zip(*batch)
            image = torch.stack(image, dim=0)
            caption2i = torch.stack(caption2i, dim=0)
            enti2attr = torch.stack(enti2attr, dim=0)
            sub2obj2rela = torch.stack(sub2obj2rela, dim=0)
            regions = torch.stack(regions, dim=0)
            boxes = torch.stack(boxes, dim=0)
            x_attr_mask = torch.zeros((batch_size, self.attr_max_x_len), dtype=torch.bool)
            x_rela_mask = torch.zeros((batch_size, self.rela_max_x_len), dtype=torch.bool)
            if self.is_sg:
                x_attr_mask = ~torch.sum(enti2attr, dim=-1).bool()
                x_rela_mask = ~torch.sum(sub2obj2rela, dim=-1).bool()

            if self.is_region:
                x_attr_mask = ~torch.sum(regions, dim=-1).bool()
            return image, caption2i, image_id, caption, enti2attr, sub2obj2rela, [x_rela_mask, x_attr_mask], regions, boxes
        return collate

    def __len__(self):
        return self.len
    
    def add_pad(self, item: torch.Tensor) -> torch.Tensor:
        padding = GlobalConfig.max_seq_len - item.shape[0] - 2
        if padding > 0:
            item = torch.cat((torch.tensor([GlobalConfig.token_bos]), item, 
                          torch.tensor([GlobalConfig.token_eos])), dim=0)
            item = torch.cat((item, torch.ones(padding, dtype=torch.int32) * GlobalConfig.padding_idx))
        elif padding < 0:
            item = item[:GlobalConfig.max_seq_len - 2]
            item = torch.cat((torch.tensor([GlobalConfig.token_bos]), item, 
                          torch.tensor([GlobalConfig.token_eos])), dim=0)
        elif padding == 0:
            item = torch.cat((torch.tensor([GlobalConfig.token_bos]), item, 
                          torch.tensor([GlobalConfig.token_eos])), dim=0)
        # item[i] < 0 is 0
        # mask = item.lt(0)
        # item[mask] = 0.
        # mask = mask.float()
        # mask = torch.cat((torch.ones(1), mask, torch.ones(1)), dim=0)
        return item
    
    def caption2vector(self, caption: str) -> torch.Tensor:
        return self.add_pad(torch.tensor(self.vocab.caption2i(caption), dtype=torch.long))

    def vector2caption(self, vector: torch.Tensor) -> str:
        return self.vocab.i2caption(vector)
    
