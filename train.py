import argparse
import torch
import numpy as np
import os
import random
import time
import torch.multiprocessing as mp
import torch.nn as nn
import json
import math

from torch import optim
from torch.nn import CrossEntropyLoss, NLLLoss
from models.ParallelModel import ParallelModel
from data.dataset.MsCocoDataset import MsCocoDataset
from data.dataset.MsCocoDataLoader import MsCocoDataLoader
from models.enums import DataType
from models.utils import print_parameter_count
from config.GlobalConfig import GlobalConfig
from torch.optim.lr_scheduler import LambdaLR
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn, TransferSpeedColumn, TaskProgressColumn
from models.utils import CalcTime
from data import evaluation
from data.evaluation.cider import Cider
from models.mutil.launch import launch
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from data.vocab.Vocab import Vocab

def get_parameters():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size', type=int, default=50)
    parse.add_argument('--epoch', type=int, default=10000)
    parse.add_argument('--lr', type=float, default=5e-8)
    parse.add_argument('--rl_lr', type=float, default=1e-6)
    parse.add_argument('--num_workers', type=int, default=0)
    parse.add_argument('--gpu_ids', type=int, default=[1], nargs='*')
    # parse.add_argument('--gpu_ids', type=list, default=[0])
    parse.add_argument('--device', type=str, default='cuda')
    parse.add_argument('--version', type=str, default='V0')
    parse.add_argument('--log_step', type=int, default=99999)
    parse.add_argument('--save_path', type=str, default='./data/output/')
    parse.add_argument('--num_heads', type=int, default=8)
    parse.add_argument('--num_encoder_layers', type=int, default=3)
    parse.add_argument('--num_decoder_layers', type=int, default=1)
    parse.add_argument('--is_rl', action='store_true', default=False)
    parse.add_argument('--only_test', action='store_true', default=False)
    parse.add_argument('--is_scheduler', action='store_true', default=False)
    # parse.add_argument('--image_path', type=str, default="/home/lab/Project/datasets/mscoco/")
    parse.add_argument('--image_path', type=str, default="/media/a1002/two/datasets/MSCOCO2014")
    # parse.add_argument('--image_path', type=str, default="F:/LAB/Dataset/Coco/")
    parse.add_argument('--annotation_path', type=str, default="./data/annotations/")
    # parse.add_argument('--hdf5_filepath', type=str, default="../X152_trainval.hdf5")
    parse.add_argument('--hdf5_filepath', type=str, default="../coco_all_align.hdf5")
    # parse.add_argument('--hdf5_filepath', type=str, default="../swin_feature.hdf5")
    # parse.add_argument('--hdf5_filepath', type=str, default="./data/features/swin_feature.hdf5")
    # parse.add_argument('--hdf5_filepath', type=str, default="F:/swin_feature.hdf5")
    args = parse.parse_args()
    return args 

def train(args):
    device = torch.device(f"{args.device}:{int(args.gpu_ids[0])}" if "cuda" in args.device else "cpu")
    vocab = Vocab()
    # Seed
    seed = 1234
    print("seed: " + str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    ##
    train_dataset = MsCocoDataset(args.image_path, args.annotation_path, DataType.TRAIN, device, hdf5_grid=args.hdf5_filepath, is_use_hdf5=True)
    # 分布式数据集
    # train_sampler = DistributedSampler(train_dataset)
    val_dataset = MsCocoDataset(args.image_path, args.annotation_path,DataType.VAL, device, hdf5_grid=args.hdf5_filepath, is_use_hdf5=True)
    test_dataset = MsCocoDataset(args.image_path, args.annotation_path,DataType.TEST, device, hdf5_grid=args.hdf5_filepath, is_use_hdf5=True)

    ##
    # model = ParallelModel(args.num_encoder_layers, args.num_decoder_layers, args.num_heads, is_use_hdf5=train_dataset.is_use_hdf5)
    model = Transformer(GlobalConfig.token_bos)
    print(f"use {args.gpu_ids} GPU!")
    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)
    print_parameter_count(model,is_simplify=True, is_print_all=False,is_print_detail=False, contain_str="decoder")
    model.to(device)
    model.train()
    # 分布式模型a
    # model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    ##
    # shuffle must be False
    train_dataLoader = MsCocoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
    # train_dataLoader = MsCocoDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, drop_last=True, num_workers=args.num_workers)
    val_dataLoader = MsCocoDataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_dataLoader = MsCocoDataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    ## nn.logSoftmax()+nn.NLLLoss()
    ## nn.Softmax()+nn.CrossEntropyLoss()
    # loss_fn = CrossEntropyLoss(ignore_index=GlobalConfig.padding_idx)
    loss_fn = NLLLoss(ignore_index=GlobalConfig.padding_idx)
    def lambda_lr(s):
        if s == 0: return 1
        print("s: " + str(s))
        s = s // 5
        lr = math.pow(0.2, s)
        if args.is_scheduler: lr = 1
        print("lr: " + str(lr * args.lr))
        return lr
    # def lambda_lr(s):
    #     s += 1
    #     return (model.d_model ** -.5) * min(s ** -.5, s * 10000 ** -1.5)

    def lambda_lr_rl(s):
        refine_epoch = 8
        if s <= refine_epoch: lr = 1
        elif s <= refine_epoch + 3: lr = 0.2
        elif s <= refine_epoch + 6: lr = 0.2 * 0.2
        else: lr = 0.2 * 0.2 * 0.2
        return lr
    use_rl = args.is_rl
    optimizer, scheduler = None, None
    cider = None
    if not use_rl:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        scheduler = LambdaLR(optimizer, lambda_lr)
    else:
        print("use rl train")
        cider = Cider()
        optimizer = optim.Adam(model.parameters(), lr=args.rl_lr, betas=(0.9, 0.98))
        scheduler = LambdaLR(optimizer, lambda_lr_rl)

    loss_val:float = 0.
    loss_val_temp:float = 0.
    start_epoch:int = 0
    min_loss_val:float = 9999.
    gen = {}
    gts = {}
    gts_val = {}
    if os.path.exists("./data/output/gts.json"):
        gts = json.load(open("./data/output/gts.json", "r"))
    if os.path.exists("./data/output/gts_val.json"):
        gts_val = json.load(open("./data/output/gts_val.json", "r"))
    #
    _save_path = os.path.join(args.save_path, "models_" + args.version + "_{}.pth")
    if use_rl:
        _save_path = os.path.join(args.save_path, "models_rl_" + args.version + "_{}.pth")
    
    __save_path = _save_path.format("best")
    if os.path.exists(__save_path):
        print("load model parameters...")
        data = torch.load(__save_path)
        start_epoch = data["epoch"] + 1
        print("data['epoch']: \t" + str(start_epoch))
        model.load_state_dict(data["parameters"], strict=True)
        optimizer.load_state_dict(data["optimizer"])
        if not args.is_scheduler:
            scheduler.load_state_dict(data["scheduler"])
        min_loss_val = data["loss"]
        print("data['loss']: \t" + str(min_loss_val))

    start = time.time()
    for epoch in range(start_epoch, args.epoch + 1):
        model.train()
        GlobalConfig.mode = DataType.TRAIN
        print("start train model...")
        with Progress(
            "|",TextColumn("[progress.description]{task.description}"),
            "|",MofNCompleteColumn(),BarColumn(bar_width=1000),TaskProgressColumn(show_speed=True),
            "|",TextColumn("[progress.loss_val]{task.fields[loss_val]}", style="color({255,0,0}})"),
            "|",TextColumn("[progress.last_loss_val]{task.fields[last_loss_val]}", style="color({0,0,255}})"),
            "|",TimeElapsedColumn(),"/",TimeRemainingColumn(),
            "|",TransferSpeedColumn(),
            "|") as progress:
            loss_val = 0.
            loss_val_temp = 0.
            scheduler.step()
            task = progress.add_task("Epoch: " + str(epoch) + " Step: 0" + str(), total=train_dataset.len, \
                loss_val="loss: " + str(round(loss_val, 5)), last_loss_val="last_loss: " + str(round(min_loss_val, 5)))
            # task = progress.add_task("Epoch: " + str(epoch) + " Step: 0" + str(), total=args.batch_size * 1, loss_val="loss: " + str(round(loss_val, 5)))
            if not args.only_test:
                if not use_rl:
                    for idx,(image,caption2i,image_id,caption,enti2attr,sub2obj2rela,sg_mask,regions,boxes) in enumerate(train_dataLoader):
                        image, caption2i = image.to(device), caption2i.to(device)
                        sg_mask = [sg_mask[0].to(device), sg_mask[1].to(device)]
                        enti2attr,sub2obj2rela = enti2attr.to(device), sub2obj2rela.to(device)
                        regions, boxes = regions.to(device), boxes.to(device)
                        optimizer.zero_grad()
                        # res, pre = model(image, caption2i, enti2attr, sub2obj2rela, image_id=image_id, sg_mask=sg_mask, regions=regions, boxes=boxes)
                        # res = model(regions, caption2i)
                        res = model(image, caption2i, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes)
                        caption2i = caption2i[:, 1:].contiguous()
                        # res = res[:, 1:, :].contiguous()
                        res = res[:, :-1, :].contiguous()
                        loss = loss_fn(res.view(-1, res.shape[2]), caption2i.view(-1))
                        loss.backward()
                        optimizer.step()
                        loss_val = loss_val * idx + loss.cpu().item()
                        loss_val = loss_val / (idx + 1)
                        loss_val_temp = loss_val_temp + loss.item()
                        if idx % args.log_step == (args.log_step - 1):
                            # _t = loss_val_temp / args.log_step
                            # print("Epoch: {} Step: {} CurrLoss: {} TotalLoss: {} Percent: {}".format(epoch, idx, round(_t, 5), round(loss_val, 5), round(idx / (train_dataset.len / args.batch_size), 5)))
                            print(train_dataset.vector2caption(pre[:1, 1:]))
                            # print(caption[0])
                            loss_val_temp = 0.
                            inter = time.time() - start
                            if inter > 3600.0:
                                start = time.time()
                                if loss_val < min_loss_val:
                                    print("start save model...")
                                    torch.save({
                                        "parameters":model.state_dict(), 
                                        "optimizer":optimizer.state_dict(),
                                        "epoch":epoch,
                                        "loss":loss_val,
                                        'scheduler': scheduler.state_dict()
                                    }, _save_path.format("time"))
                                    print("end save model...")
                        # if not progress.finished:
                        progress.update(task, advance=args.batch_size, loss_val="loss: " + str(round(loss_val, 5)), description="Epoch: " + str(epoch) + " Step: " + str(idx))
                else:
                    for idx,(image,caption2i,image_id,caption,enti2attr,sub2obj2rela) in enumerate(train_dataLoader):
                        image, caption2i = image.to(device), caption2i.to(device)
                        optimizer.zero_grad()
                        # Rewards
                        res = model.step(image, caption2i, enti2attr,sub2obj2rela, image_id=image_id)
                        caps_gen = train_dataset.vocab.i2caption(res)
                        reward = cider.compute_score({0: caption}, {0: caps_gen})[1].astype(np.float32)
                        reward = torch.from_numpy(reward).to(device)
                        reward_baseline = torch.mean(reward, -1, keepdim=True)
                        loss = -torch.mean(res, -1) * (reward - reward_baseline)
                        loss = loss.mean()
                        loss.backward()
                        optimizer.step()
                        loss_val = loss_val * idx + loss.cpu().item()
                        loss_val = loss_val / (idx + 1)
                        # if not progress.finished:
                        progress.update(task, advance=args.batch_size, loss_val="loss: " + str(round(loss_val, 5)), description="Epoch: " + str(epoch) + " Step: " + str(idx))
                print("end train model...")
                if loss_val < min_loss_val:
                    print("start save model...")
                    # torch.save({
                    #     "parameters":model.state_dict(), 
                    #     "optimizer":optimizer.state_dict(),
                    #     "epoch":epoch,
                    #     "loss":loss_val,
                    #     'scheduler': scheduler.state_dict()
                    # }, _save_path.format(epoch))
                    torch.save({
                        "parameters":model.state_dict(), 
                        "optimizer":optimizer.state_dict(),
                        "epoch":epoch,
                        "loss":loss_val,
                        'scheduler': scheduler.state_dict()
                    }, _save_path.format("best"))
                    print("end save model...")
                    min_loss_val = loss_val
                    progress.update(task, advance=0., min_loss_val="last_loss: " + str(round(min_loss_val, 5)))
                else:
                    print("not save model...")
        # ----------------------------------------------------------------------------------------------------------------
        gen = {}
        model.eval()
        GlobalConfig.mode = DataType.VAL
        print("start validate model...")
        with Progress() as progress:
            task = progress.add_task("Epoch:", total=val_dataset.len)
            print("start generate caption...")
            for _,(image,caption,image_id,enti2attr,sub2obj2rela,sg_mask,regions,boxes) in enumerate(val_dataLoader):
                with torch.no_grad():
                    image = image.to(device)
                    sg_mask = [sg_mask[0].to(device), sg_mask[1].to(device)]
                    enti2attr, sub2obj2rela = enti2attr.to(device), sub2obj2rela.to(device)
                    regions, boxes = regions.to(device), boxes.to(device)
                    # res, _ = model(image, image_id=image_id,enti2attr=enti2attr,sub2obj2rela=sub2obj2rela,sg_mask=sg_mask,regions=regions,boxes=boxes)
                    res, _ = model.beam_search([image, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes], GlobalConfig.max_seq_len, GlobalConfig.token_eos, 5, 1)
                    # gen[image_id[0]] = [res[0]]
                    gen[image_id[0]] = vocab.i2caption(res)
                    # gen[image_id[0]] = res
                for i, _id in enumerate(image_id):
                    gts_val[_id] = caption[i]
                # if not progress.finished:
                progress.update(task, advance=1)
        json.dump(gen, open("./data/output/gen_val_{}.json".format(args.version), "w"))
        json.dump(gts_val, open("./data/output/gts_val.json", "w"))
        print("end validate model...")
        print("start calc score")
        # gen = json.load(open("./data/output/gen_val_{}.json".format(args.version), "r"))
        score, scores = evaluation.compute_scores(gts_val, gen, False)
        print("score:" + str(score))
        # print(scores)
        cider = scores["CIDEr"].tolist()
        cider.insert(0, score["CIDEr"])
        json.dump({"CIDEr": cider}, open("./data/output/gen_val_{}_scores.json".format(args.version), "w"))
        # ----------------------------------------------------------------------------------------------------------------
        gen = {}
        GlobalConfig.mode = DataType.TEST
        print("start test model...")
        with Progress() as progress:
            task = progress.add_task("Epoch:", total=test_dataset.len)
            print("start generate caption...")
            for _,(image,caption,image_id,enti2attr,sub2obj2rela,sg_mask,regions,boxes) in enumerate(test_dataLoader):
                with torch.no_grad():
                    image = image.to(device)
                    sg_mask = [sg_mask[0].to(device), sg_mask[1].to(device)]
                    enti2attr,sub2obj2rela = enti2attr.to(device), sub2obj2rela.to(device)
                    regions, boxes = regions.to(device), boxes.to(device)
                    # res, _ = model(image, image_id=image_id,enti2attr=enti2attr,sub2obj2rela=sub2obj2rela,sg_mask=sg_mask,regions=regions,boxes=boxes)
                    # res, _ = model.beam_search(regions, GlobalConfig.max_seq_len, GlobalConfig.token_eos, 5, 1)
                    res, _ = model.beam_search([image, image_id, enti2attr, sub2obj2rela, sg_mask, regions, boxes], GlobalConfig.max_seq_len, GlobalConfig.token_eos, 5, 1)
                    # gen[image_id[0]] = [res[0]]
                    gen[image_id[0]] = vocab.i2caption(res)
                    # gen[image_id[0]] = res
                # for i, _id in enumerate(image_id):
                #     gts[_id] = caption[i]
                # if not progress.finished:
                progress.update(task, advance=1)
        json.dump(gen, open("./data/output/gen_test_{}.json".format(args.version), "w"))
        print("end test model...")
        print("start calc score")
        score, scores = evaluation.compute_scores(gts, gen, False)
        print("score:" + str(score))
        # print(scores)
        cider = scores["CIDEr"].tolist()
        cider.insert(0, score["CIDEr"])
        json.dump({"CIDEr": cider}, open("./data/output/gen_test_{}_scores.json".format(args.version), "w"))

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '2.0'
    # torch.set_printoptions(threshold=np.inf)
    args = get_parameters()
    train(args)
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29501"
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ[
    #     "TORCH_DISTRIBUTED_DEBUG"
    # ] = "DETAIL"  # set to DETAIL for runtime logging.
    # mp.spawn(train, nprocs=2, args=(args,))
    # launch(train,2,1,0,"auto",(args,))
