import os
import numpy as np

# 
sg_root = "data/features/coco_pred_sg"
sg_dict_path = "data/features/coco_pred_sg_rela.npy"

# sg
image_id = 9
sg = np.load(os.path.join(sg_root, f"{image_id}.npy"), allow_pickle=True, encoding="latin1")[()]
sg_rela = sg['rela_matrix']
sg_attr = sg['obj_attr']

# dict
sg_dict = np.load(sg_dict_path, allow_pickle=True, encoding="latin1")[()]
sg_i2w = sg_dict["i2w"]
sg_rela_dict = sg_dict["rela_dict"]
sg_w2i = sg_dict["w2i"]

sg_obj = []
print(sg_attr)
for i, obj, attr in sg_attr:
    obj = sg_i2w[obj]
    sg_obj.append(obj)
    attr = sg_i2w[attr]
    print(f"{attr}-{obj}")

print(sg_rela)
for i, j, rela in sg_rela:
    sub = sg_obj[int(i)]
    rela = sg_rela_dict[rela]
    attr = sg_obj[int(j)]
    print(f"{sub}-{rela}-{obj}")
