import torch
import argparse
import os
import json
import torch.nn as nn

from data.dataset.MsCocoDataLoader import MsCocoDataLoader
from data.dataset.MsCocoDataset import MsCocoDataset
from models.ParallelModel import ParallelModel
from models.enums import DataType
from rich.progress import Progress
from data import evaluation
from config.GlobalConfig import GlobalConfig
from models.enums import DataType

def get_parameters():
    parse = argparse.ArgumentParser()
    parse.add_argument('--num_workers', type=int, default=0)
    parse.add_argument('--device', type=str, default='cuda')
    parse.add_argument('--gpu_ids', type=int, default=[1], nargs='*')
    parse.add_argument('--save_path', type=str, default='./data/output/')
    parse.add_argument('--log_path', type=str, default='./log')
    parse.add_argument('--version', type=str, default='V_1')
    parse.add_argument('--is_generate', action='store_false', default=True)
    parse.add_argument('--num_heads', type=int, default=4)
    parse.add_argument('--num_encoder_layers', type=int, default=1)
    parse.add_argument('--num_decoder_layers', type=int, default=1)
    parse.add_argument('--image_path', type=str, default="/home/lab/Project/datasets/mscoco/")
    # parse.add_argument('--image_path', type=str, default="F:/LAB/Dataset/Coco/")
    parse.add_argument('--annotation_path', type=str, default="./data/annotations/")
    parse.add_argument('--hdf5_filepath', type=str, default="../X152_trainval.hdf5")
    # parse.add_argument('--hdf5_filepath', type=str, default="../swin_feature.hdf5")
    # parse.add_argument('--hdf5_filepath', type=str, default="./data/features/test.pth")
    # parse.add_argument('--hdf5_filepath', type=str, default="F:/swin_feature.hdf5")
    args = parse.parse_args()
    return args

def generate_caption(args):
    device = torch.device(f"{args.device}:{int(args.gpu_ids[0])}" if "cuda" in args.device else "cpu")
    GlobalConfig.mode = DataType.TEST
    ##
    test_dataset = MsCocoDataset(args.image_path, args.annotation_path, DataType.TEST, device, hdf5_grid=args.hdf5_filepath, is_use_hdf5=True)
    test_dataLoader = MsCocoDataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = ParallelModel(args.num_encoder_layers, args.num_decoder_layers, args.num_heads, is_use_hdf5=test_dataset.is_use_hdf5)
    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    model.eval()
    model.to(device)
    ##
    save_path = os.path.join(args.save_path, "models_"  + args.version +  ".pth")
    if os.path.exists(save_path):
        print("load model parameters...")
        data = torch.load(save_path)
        if "parameters" in data:
            model.load_state_dict(data["parameters"], strict=True)
    ##
    gen = {}
    gts = {}
    if os.path.exists("./data/output/gts.json"):
        gts = json.load(open("./data/output/gts.json", "r"))
    if os.path.exists("./data/output/gen_{}.json".format(args.version)) and not args.is_generate:
        gen = json.load(open("./data/output/gen_{}.json".format(args.version), "r"))
    else:
        with Progress() as progress:
            task = progress.add_task("Epoch:", total=test_dataset.len)
            print("start generate test caption...")
            for _,(image,caption,image_id,enti2attr,sub2obj2rela,sg_mask,regions,boxes) in enumerate(test_dataLoader):
                with torch.no_grad():
                    image = image.to(device)
                    sg_mask = [sg_mask[0].to(device), sg_mask[1].to(device)]
                    res, _ = model(image, image_id=image_id,enti2attr=enti2attr,sub2obj2rela=sub2obj2rela,sg_mask=sg_mask,regions=regions,boxes=boxes)
                    # print(res)
                    gen[image_id[0]] = [res[0]]
                    # gen[image_id[0]] = res
                # for i, _id in enumerate(image_id):
                #     gts[_id] = caption[i]
                if not progress.finished:
                    progress.update(task, advance=1)
        json.dump(gen, open("./data/output/gen_{}.json".format(args.version), "w"))
        # json.dump(gts, open("./data/output/gts.json", "w"))
    print("start calc score")
    score, scores = evaluation.compute_scores(gts, gen, False)
    print("score:" + str(score))
    # print(scores)
    cider = scores["CIDEr"].tolist()
    cider.insert(0, score["CIDEr"])
    json.dump({"CIDEr": cider}, open("./data/output/gen_{}_scores.json".format(args.version), "w"))
    # print("start calc score with spice")
    # scores, _ = evaluation.compute_scores(gts, gen, True)
    # print("scores:" + str(scores))

if __name__ == "__main__":
    args = get_parameters()
    generate_caption(args)
