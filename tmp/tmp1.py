#first reproduce training file to see the whole logic

import os
import os.path as osp
import torch 
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
import numpy as np
import logging

from apex import amp
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch_da, crop_image
from utils.utils import set_random_seed, resume_from_ckpt, mkdir
from utils.transforms import build_transforms
from models.seqnet_da import SeqNetDa

from datasets import build_test_loader, build_train_loader_da, build_dataset,build_train_loader_da_dy_cluster,build_cluster_loader

from spcl.models.dsbn import convert_dsbn
from spcl.models.hm import HybridMemory
from spcl.evaluators import Evaluator, extract_features,extract_dy_features

# 配置日志记录器
logging.basicConfig(filename='logfile.log', level=logging.INFO, format='%(asctime)s - %(message)s',encoding='utf-8')

def main(args):
    #得到train时的各项参数
    cfg=get_default_cfg()
    if args.cfg_file:
        #与yaml中参数合并
        cfg.merge_from_file(args.cfg_file)
    #合并命令行中参数
    cfg.merge_from_list(args.opt)
    #冻结参数
    cfg.freeze()

    device=torch.device(cfg.DEVICE)
    if cfg.SEED>=0:
        set_random_seed(cfg.SEED) #把可能随机的地方种子都设置成一样的

    logging.info("source dataset: ",cfg.INPUT.DATASET)
    logging.info("target dataset: ",cfg.INPUT.DATASET)
    logging.info("\nCreating model and convert dsbn")

    model=SeqNetDa(cfg)
    #将模型域归一化
    convert_dsbn(model.roi_heads.reid_head)
    model.to(device)

    logging.info("building dataset")
    transforms = build_transforms(is_train=False)
    dataset_source_train=build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train", is_source=True)
    source_classes=dataset_source_train.num_train_pids
    logging.info("source classes :"+str(source_classes))

    logging.info("loading test data")
    #在gallery中找query
    gallery_loader,query_loader=build_test_loader(cfg)

    if args.eval:
        logging.warning(args.ckpt, "--ckpt must be specified when --eval enabled")
        resume_from_ckpt(args.ckpt,model)
        dataset_target_train=build_dataset(cfg.INPUT.TDATASET,cfg.INPUT.TDATA_ROOT, transforms,"train", is_source=False)
        tgt_cluster_loader=build_cluster_loader(cfg,dataset_target_train)
        model.eval()
        evaluate_performance(
            model,
            gallery_loader, 
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
            )
        exit(0) #eval完了就退出

    memory=HybridMemory(256,source_classes,source_classes,temp=0.05,momentum=0.2).to(device)

    # init source domian identity level centroid
    logging.info("==> Initialize source-domain class centroids in the hybrid memory")
    sour_cluster_loader = build_cluster_loader(cfg,dataset_source_train)
    #提取特征
    sour_fea_dict = extract_dy_features(cfg, model, sour_cluster_loader, device, is_source=True)
    #计算特征中心
    source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    source_centers = torch.stack(source_centers,0)
    source_centers = F.normalize(source_centers, dim=1)
    logging.info("source_centers length")
    logging.info(len(source_centers))
    logging.info(source_centers.shape)
    logging.info("the last one is the feature of 5555, remember don't use it")
    #更新记忆库
    memory.features = source_centers.cuda()
    del source_centers, sour_fea_dict, sour_cluster_loader

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    model.roi_heads.memory = memory
    #自动混合精度优化
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    logging.info("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    target_start_epoch = cfg.TARGET_REID_START
    with open(path, "w") as f:
        f.write(cfg.dump())
    logging.info(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        logging.info(f"TensorBoard files are saved to {tf_log_path}")


