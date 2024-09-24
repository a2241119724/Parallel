import torch
import argparse
import os
import json

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
    parse.add_argument('--batch_size', type=int, default=1)
    parse.add_argument('--num_workers', type=int, default=0)
    parse.add_argument('--device', type=str, default='cuda')
    parse.add_argument('--save_path', type=str, default='./data/output/')
    parse.add_argument('--log_path', type=str, default='./log')
    parse.add_argument('--version', type=str, default='V-1')
    parse.add_argument('--is_generate', action='store_true', default=False)
    parse.add_argument('--num_heads', type=int, default=8)
    parse.add_argument('--num_encoder_layers', type=int, default=1)
    parse.add_argument('--num_decoder_layers', type=int, default=1)
    parse.add_argument('--image_path', type=str, default="/home/lab/Project/datasets/mscoco/")
    parse.add_argument('--annotation_path', type=str, default="./data/annotations/")
    parse.add_argument('--hdf5_filepath', type=str, default="../swin_feature.hdf5")
    # parse.add_argument('--hdf5_filepath', type=str, default="F:/swin_feature.hdf5")
    args = parse.parse_args()
    return args

def generate_caption(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_batch_size = args.batch_size
    GlobalConfig.mode = DataType.TEST
    ##
    # train_dataset = MsCocoDataset(args.image_path, args.annotation_path, DataType.TRAIN, hdf5_filepath=args.hdf5_filepath)
    # val_dataset = MsCocoDataset(args.image_path, args.annotation_path,DataType.VAL,hdf5_filepath=args.hdf5_filepath)
    test_dataset = MsCocoDataset(args.image_path, args.annotation_path,DataType.TEST,hdf5_filepath=args.hdf5_filepath)
    # train_dataLoader = MsCocoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_dataLoader = MsCocoDataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataLoader = MsCocoDataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = ParallelModel(test_batch_size, args.num_encoder_layers, args.num_decoder_layers, device)
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
    with Progress() as progress:
        task = progress.add_task("Epoch:", total=test_dataset.len)
        print("start generate test caption...")
        for _,(image,caption,image_id,enti2attr, sub2rela2obj) in enumerate(test_dataLoader):
            with torch.no_grad():
                image = image.to(device)
                res, _ = model(image, image_id=image_id,enti2attr=enti2attr,sub2rela2obj=sub2rela2obj)
                print({image_id[0] : res})
                print({image_id[0] : caption[0]})
            progress.update(task, advance=test_batch_size)

if __name__ == "__main__":
    args = get_parameters()
    generate_caption(args)
