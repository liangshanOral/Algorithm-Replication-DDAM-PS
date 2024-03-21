from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from models.oim import OIMLoss
from models.resnet import build_resnet
from models.rpn_da import RegionProposalNetworkDA
from models.roi_head_da import SeqRoIHeadsDa
from models.da_head import DomainAdaptationModule
from apex import amp
from models.idm_module import IDM
from models.idm_loss import BridgeProbLoss,BridgeFeatLoss,DivLoss

class SeqNetDa(nn.Module):
    def __init__(self,cfg):
        super(SeqNetDa, self).__init__()
        self.target_start_epoch=cfg.TARGET_REID_START
        #最先输入的网络
        backbone,box_head=build_resnet(name="resnet50",pretrained=True)
        #生成不同尺寸和长宽比的锚框
        anchor_generator=AnchorGenerator(sizes=((32,64,128,256,512),),aspect_ratios=((0.5,1.0,2.0),))
        #输入特征，每个位置锚框数
        head=RPNHead(in_channels=backbone.out_channels,num_anchors=anchor_generator.num_anchors_per_location()[0],)
        #training时保留的通常更多
        pre_nms_top_n=dict(training=cfg.MODEL.RPN.PRE_NUM_TOPN_TRAIN,testing=cfg.MODEL.RPN.PRE_NUMS_TOPN_TEST)
        post_nms_top_n = dict(training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST)
        rpn=RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MoDEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN, #负样本IoU阈值
            batch_size_per_image=cfg.MODEL.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN, #正负样本比例
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NUS_THRESH, #非极大值抑制
        )

        faster_rcnn_predictor=FastRCNNPredictor(2048,2)
        reid_head=deepcopy(box_head)
        box_roi_pool=MultiScaleRoIAlign(featmap_names=["feat_res4"],output_size=14,sampling_ratio=2)
        box_predictor=BBoxRegressor(2048,num_classes=2,bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        roi_heads = SeqRoIHeadsDa(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )
        #对图像预处理
        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        #step1
        self.backbone = backbone
        #step2
        self.rpn = rpn
        self.roi_heads = roi_heads
        # here modified, for adapting to amp
        self.roi_heads.box_roi_pool.forward = amp.half_function(self.roi_heads.box_roi_pool.forward)
        self.transform = transform
        self.da_heads = DomainAdaptationModule(cfg.SOLVER.LW_DA_INS)

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID
        self.lw_box_reid_t = cfg.SOLVER.LW_BOX_REID_T
        self.IDM=IDM(1024)

        #IDM losses:
        self.criterion_ce = BridgeProbLoss().cuda()
        self.criterion_bridge_feat = BridgeFeatLoss()
        self.criterion_diverse = DivLoss()


class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas
