import torch
import json
import numpy as np
import os
import h5py
import sys

sys.path.append(os.path.abspath("./"))

from rich.progress import track
from data.vocab.Vocab import Vocab

def preprocess_sg():
    image_id = []
    test = json.load(open("data/annotations/coco_lab_karpathy_test2014.json", "r"))
    train = json.load(open("data/annotations/coco_lab_karpathy_train2014.json", "r"))
    val = json.load(open("data/annotations/coco_lab_karpathy_val2014.json", "r"))
    
    vocab2i = Vocab().vocab2i_fn
    
    sg_dict_path = "data/features/coco_pred_sg_rela.npy"
    sg_dict = np.load(sg_dict_path, allow_pickle=True, encoding="latin1")[()]
    sg_i2w = sg_dict["i2w"]
    # sg_rela_dict = sg_dict["rela_dict"]
    # sg_w2i = sg_dict["w2i"]

    for it in test:
        image_id.append(it["image_id"])
        
    for it in train:
        image_id.append(it["image_id"])

    for it in val:
        image_id.append(it["image_id"])
        
    image_id = list(set(image_id))

    sg_root = "data/features/coco_pred_sg"

    # h5 = h5py.File("data/features/sg_str.h5", "w")

    # attr_len = 0
    # rela_len = 0
    # enti_len = 0

    for _id in track(image_id, description="sg: "):
        sg = np.load(os.path.join(sg_root, f"{_id}.npy"), allow_pickle=True, encoding="latin1")[()]
        rela_matrix = sg['rela_matrix']
        obj_attr = sg['obj_attr']
        
        _rela_matrix = []
        _obj_attr = []
        
        sg_obj = []
        for _, obj, attr in obj_attr:
            # _obj = sg_i2w[obj]
            # sg_obj.append(_obj)
            # _obj_attr.append([int(obj),int(attr)])
            obj = int(vocab2i(sg_i2w[obj]))
            attr = int(vocab2i(sg_i2w[attr]))
            sg_obj.append(obj)
            _obj_attr.append([obj, attr])

        # __rela_matrix = []
        for sub, obj, rela in rela_matrix:
            # sub = int(sub)
            # obj = int(obj)
            # _rela_matrix.append([sub, obj, int(rela)])
            # if f"{str(sg_obj[sub])}-{str(sg_obj[obj])}" not in __rela_matrix:
            #     __rela_matrix.append(f"{str(sg_obj[sub])}-{str(sg_obj[obj])}")
            #     _rela_matrix.append([sub, obj, int(rela)])
            sub = sg_obj[int(sub)]
            obj = sg_obj[int(obj)]
            rela = int(rela)
            _rela_matrix.append([sub, obj, rela])

        # ___rela_matrix = []
        # for sub, obj, rela in _rela_matrix:
        #     if sg_obj[sub] not in ___rela_matrix:
        #         ___rela_matrix.append(sg_obj[sub])
        #     if sg_obj[obj] not in ___rela_matrix:
        #         ___rela_matrix.append(sg_obj[obj])
                
        # ___rela_matrix = []
        # for sub, obj, rela in _rela_matrix:
        #     if sub not in ___rela_matrix:
        #         ___rela_matrix.append(sub)
        #     if obj not in ___rela_matrix:
        #         ___rela_matrix.append(obj)

        # if len(___rela_matrix) > enti_len:
        #     enti_len = len(___rela_matrix)
        # if len(_rela_matrix) > rela_len:
        #     rela_len = len(_rela_matrix)
        # if len(_obj_attr) > attr_len:
        #     attr_len = len(_obj_attr)
        
        # h5[f"{_id}_rela_matrix"] = _rela_matrix
        # h5[f"{_id}_obj_attr"] = _obj_attr

    # h5.close()

    # print(f"attr_len: {attr_len}")
    # print(f"enti_len: {enti_len}")
    # print(f"rela_len: {rela_len}")

if __name__ == "__main__":
    preprocess_sg()
