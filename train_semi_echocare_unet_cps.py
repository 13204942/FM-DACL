# autodl-tmp/Mamba-UNet/code/train_fully_supervised_2D_ViT.py
import argparse
import logging
import os
import random
import shutil
import sys
import time
import cv2
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
import albumentations as A
from tqdm import tqdm
# from config import get_config
# from dataloaders import utils
from dataset.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, WeakStrongAugment)
from dataset.fetus import FETUSSemiDataset
from model.unet import UNet
from model.Echocare import Echocare_UniMatch
from model.cnn import CNN
# from networks.projector import projectors as proj
# from networks.net_factory import net_factory
# from utils import losses, ramps #, metrics
# from utils.dice import DSC
from util import ramps
from util.metrics import dice_score, dice
# from utils.losses import ConstraLoss, ConstraLoss_AvgProj, info_nce_loss, hd_loss, softmax_kl_loss, softmax_mse_loss, ConstraLoss_multi_AvgProj, smooth_l1_loss, global_contra_loss, Uncertainty_Loss
from util.utils import (
    AverageMeter,
    DiceLoss,
    apply_view_mask_logits,
    build_allowed_mat,
    build_seg_allowed_mat,
    build_same_view_perm,
    compute_pos_weight_from_loader,
    count_params,
    load_pretrained_flexible,
    log_train_tb,
    log_val_perclass_tb,
    log_val_tb,
    masked_bce_with_logits,
    masked_metrics_with_threshold_search,
    masked_mse,
    nsd_binary,
    update_meters,
)

torch.backends.cuda.matmul.allow_tf32 = False  # 禁止矩阵乘法使用tf32
torch.backends.cudnn.allow_tf32 = False        # 禁止卷积使用tf32

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Semi_Mamba_UNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=3,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[448, 448],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1234, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    # '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )

parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# FETUS
parser.add_argument("--ssl_ckpt", type=str, default='model/pretrain/echocare_encoder.pth')  # for echocare
parser.add_argument("--seg_num_classes", type=int, default=15)
parser.add_argument("--cls_num_classes", type=int, default=7)
parser.add_argument("--view_num_classes", type=int, default=4)
parser.add_argument("--train_labeled_json", type=str, default="data/train_labeled.json")
parser.add_argument("--train_unlabeled_json", type=str, default="data/train_unlabeled.json")
parser.add_argument("--valid_labeled_json", type=str, default="data/valid.json")
parser.add_argument("--resize_target", type=int, default=256)
# Configurable masks / thresholds / weights (JSON string or JSON file path)
parser.add_argument(
    "--seg-allowed",
    type=str,
    default=None,
    help='JSON string or .json path for segmentation allowed mapping, e.g. \'{"0":[0,1], "1":[0,2]}\'',
)
parser.add_argument(
    "--cls-allowed",
    type=str,
    default=None,
    help='JSON string or .json path for classification allowed mapping, e.g. \'{"0":[0,1], "1":[2,3]}\'',
)

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=391,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--contrastive', type=float,
                    default=5.0, help='contrastive consistency')
parser.add_argument('--cps', type=float,
                    default=5.0, help='cps')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--ict_alpha', type=float, default=0.5,
                    help='ict_alpha')
args = parser.parse_args()
# config = get_config(args)
# print(config)

device_name = "cuda:0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_SEG_ALLOWED = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],           # 4CH
    1: [0, 1, 2, 4, 8],                    # LVOT
    2: [0, 6, 8, 9, 10, 11, 12],           # RVOT
    3: [0, 9, 12, 13, 14],                 # 3VT
}

DEFAULT_CLS_ALLOWED = {
    0: [0, 1],
    1: [0, 2, 3],
    2: [4, 5],
    3: [2, 5, 6],
}

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def get_current_consistency_weight(epoch, weight):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def discrepancy_calc_2(v1, v2):
    n, c, h, w = v1.size()    
    kl = torch.sum(v1 * torch.log((v1 + 1e-8) / (v2 + 1e-8)), dim=1)  # [B, H, W]

    H_p1 = -torch.sum(v1 * torch.log(v1 + 1e-8), dim=1)  # entropy
    H_p2 = -torch.sum(v2 * torch.log(v2 + 1e-8), dim=1)
    H_joint = -torch.sum(0.5*(v1+v2) * torch.log(0.5*(v1+v2) + 1e-8), dim=1)
    loss = (H_p1 + H_p2 - H_joint).mean() + kl.mean()
    return loss


def discrepancy_calc(v1, v2):
    """
    dis_loss for two different classifiers
    input : v1,v2
    output : discrepancy
    """
    assert v1.dim() == 4 # [B, C, H, W]
    assert v2.dim() == 4 # [B, C, H, W]
    n, c, h, w = v1.size()
    inner = torch.mul(v1, v2)
    print(inner.shape)
    v1 = v1.permute(2, 3, 1, 0)
    v2 = v2.permute(2, 3, 0, 1)
    mul = v1.matmul(v2)
    mul = mul.permute(2, 3, 0, 1)
    print(mul.shape)
    dis = torch.sum(mul) - torch.sum(inner)
    print(torch.sum(mul))
    print(torch.sum(inner))
    print(dis)
    dis = dis / (h * w)
    print(f'discrepancy_calc: {dis}')
    return dis


def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

        
def _load_json_arg(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    with open(s, "r", encoding="utf-8") as f:
        return json.load(f)        
        
def load_allowed_mapping(value: Optional[str], default: Dict[int, List[int]]) -> Dict[int, List[int]]:
    raw = _load_json_arg(value)
    if raw is None:
        return default
    if not isinstance(raw, dict):
        raise ValueError("Allowed mapping must be a JSON object: {view_id: [class_ids...]}")
    out: Dict[int, List[int]] = {}
    for k, v in raw.items():
        kk = int(k)
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"Allowed[{k}] must be a list of ints.")
        out[kk] = [int(x) for x in v]
    return out


def build_optimizer(model_name, model):
    if model_name == "unet":
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
        base_lrs = [args.base_lr]   # Only 1 group
        return optimizer, base_lrs
    
    if model_name == "cnn":
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
        base_lrs = [args.base_lr]   # Only 1 group
        return optimizer, base_lrs    

    if model_name == "echocare":
        base_backbone_lr = 1e-4
        base_head_lr = 1e-3

        backbone_params, head_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "Swin_encoder" in n:      # You defined encoder as backbone
                backbone_params.append(p)
            else:
                head_params.append(p)
        print("backbone params:", sum(p.numel() for p in backbone_params)/1e6, "M")
        print("head params:", sum(p.numel() for p in head_params)/1e6, "M")

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": base_backbone_lr},
                {"params": head_params, "lr": base_head_lr},
            ],
            weight_decay=0.01,
        )
        base_lrs = [base_backbone_lr, base_head_lr]
        return optimizer, base_lrs

    raise ValueError(f"Unknown opt preset: {opt_name}")

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    labeled_slice = args.labeled_num
    val_size = 50

    seg_allowed = load_allowed_mapping(args.seg_allowed, DEFAULT_SEG_ALLOWED)
    cls_allowed = load_allowed_mapping(args.cls_allowed, DEFAULT_CLS_ALLOWED)
    
    allowed_seg_mat = build_seg_allowed_mat(device, seg_allowed, args.view_num_classes, args.seg_num_classes)
    allowed_cls_mat = build_allowed_mat(device, cls_allowed, num_views=args.view_num_classes, num_classes=args.cls_num_classes)    

    con_rampup = args.consistency_rampup
    # iter_num//100
    rampup_rate =  int((labeled_slice / args.labeled_bs) * 400 / con_rampup)
    
    model1 = Echocare_UniMatch(
            in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            view_num_classes=args.view_num_classes,
            ssl_checkpoint=args.ssl_ckpt,  # You need to add --ssl-ckpt in args
            ).to(device)

    # model2 = UNet(in_chns=1,
    #         seg_class_num=args.seg_num_classes,
    #         cls_class_num=args.cls_num_classes,
    #         view_num_classes=args.view_num_classes,
    #         ).to(device)
    
    model2 = CNN(in_chns=1,
            seg_class_num=args.seg_num_classes,
            cls_class_num=args.cls_num_classes,
            ).to(device)    
    
    # ema_model = UNet(in_chns=1,
    #         seg_class_num=args.seg_num_classes,
    #         cls_class_num=args.cls_num_classes,
    #         view_num_classes=args.view_num_classes,
    #         ).to(device)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    # tr_transforms = A.Compose(
    #     [
    #         A.Rotate(limit=20, p=0.5),
    #         A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),contrast_limit=(-0.5, 0.5)),
    #         A.Blur(blur_limit=(3, 3), p=0.3),
    #         A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
    #      ],
    # )
    
    # tensor_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((args.patch_size[0], args.patch_size[1])),
    # ])  
    
    # FETUS dataset
    db_train_u = FETUSSemiDataset(args.train_unlabeled_json, "train_u", size=args.resize_target)
    db_train_l = FETUSSemiDataset(args.train_labeled_json, "train_l", size=args.resize_target)
    
    db_train = ConcatDataset([db_train_l, db_train_u])
    logging.info(f'total lenght of train data: {len(db_train)}')

    db_valid = FETUSSemiDataset(args.valid_labeled_json, "valid", size=args.resize_target)
    logging.info(f'total lenght of valid data: {len(db_valid)}')

    # train_loader_l = DataLoader(db_train_l, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    # train_loader_u = DataLoader(db_train_u, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    # valid_loader = DataLoader(db_valid, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)    

    # samples = random.sample(range(1, len(db_val)), val_size)
    # db_val = Subset(db_val, samples)

    total_slices = len(db_train)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    
    # print("labeled_idxs: {}, unlabeled_idxs: {}".format(labeled_idxs, unlabeled_idxs)) 

    train_loader_l = DataLoader(db_train_l, batch_size=1, shuffle=True, num_workers=1)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_valid, batch_size=1, pin_memory=True, shuffle=False, num_workers=1)
    
    pos_weight = compute_pos_weight_from_loader(train_loader_l, allowed_cls_mat, args.cls_num_classes, device).to(device)
    
    model1.train()
    model2.train()
    
    # optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
    #                        momentum=0.9, weight_decay=0.0001)
    # optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
    #                        momentum=0.9, weight_decay=0.0001)
    optimizer1, _ = build_optimizer('echocare', model1)
    optimizer2, _ = build_optimizer('unet', model2)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=args.seg_num_classes)    

    time1 = int(time.time())
    writer = SummaryWriter(snapshot_path + f'/log_{time1}')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, ((image_x, view_x, mask_x, class_label_x)) in enumerate(trainloader):
            image_x = image_x.to(device)
            view_x = view_x.to(device).long().view(-1)
            mask_x = mask_x.to(device)
            class_label_x = class_label_x.to(device)
            
            mask_x_allowed = allowed_cls_mat[view_x]            
            
            labeled_image_x = image_x[:args.labeled_bs]
            unlabeled_image_x = image_x[args.labeled_bs:]
            # noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            # ema_inputs = unlabeled_volume_batch + noise
            
            out1, out1_cls = model1(image_x)
            out1 = apply_view_mask_logits(out1, view_x, allowed_seg_mat)  # use_hard_view_mask          
            out1_soft1 = torch.softmax(out1, dim=1)
            
            out2, out2_cls = model2(image_x) 
            out2 = apply_view_mask_logits(out2, view_x, allowed_seg_mat)  # use_hard_view_mask
            out2_soft2 = torch.softmax(out2, dim=1)
            
            if iter_num < 2000:
                # consistency_loss1 = 0.0
                u_mix_cls_loss2 = 0.0
                consistency_loss2 = 0.0

            consistency_weight_cps = get_current_consistency_weight(iter_num // rampup_rate, args.cps)
            consistency_weight_mt = get_current_consistency_weight(iter_num // rampup_rate, args.consistency)
            contra_weight_contra = get_current_consistency_weight(iter_num // rampup_rate, args.contrastive)
            
            loss1 = 0.5 * (ce_loss(out1[:args.labeled_bs], mask_x[:args.labeled_bs])+ 
                           dice_loss(out1_soft1[:args.labeled_bs], mask_x[:args.labeled_bs].unsqueeze(1).float()))
            # loss1_cls = F.binary_cross_entropy_with_logits(out1_cls[:args.labeled_bs], class_label_x[:args.labeled_bs], reduction="none", pos_weight=pos_weight)
            loss1_cls = masked_bce_with_logits(out1_cls[:args.labeled_bs], class_label_x[:args.labeled_bs].float(), mask_x_allowed[:args.labeled_bs], pos_weight=pos_weight)
            
            loss2 = 0.5 * (ce_loss(out2[:args.labeled_bs], mask_x[:args.labeled_bs]) + 
                           dice_loss(out2_soft2[:args.labeled_bs], mask_x[:args.labeled_bs].unsqueeze(1).float()))
            # loss2_cls = F.binary_cross_entropy_with_logits(out2_cls[:args.labeled_bs], class_label_x[:args.labeled_bs], reduction="none", pos_weight=pos_weight)
            loss2_cls = masked_bce_with_logits(out2_cls[:args.labeled_bs], class_label_x[:args.labeled_bs].float(), mask_x_allowed[:args.labeled_bs], pos_weight=pos_weight)
            
            pseudo_outputs1 = torch.argmax(out1_soft1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(out2_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs1_cls = torch.sigmoid(out1_cls[args.labeled_bs:])
            pseudo_outputs2_cls = torch.sigmoid(out2_cls[args.labeled_bs:])

            pseudo_supervision1 = dice_loss(out1_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            # pseudo_supervision1_cls = F.binary_cross_entropy_with_logits(out1_cls[args.labeled_bs:], pseudo_outputs2_cls, reduction="none", pos_weight=pos_weight)
            pseudo_supervision1_cls = masked_bce_with_logits(out1_cls[args.labeled_bs:], 
                                                             pseudo_outputs2_cls.float(), 
                                                             mask_x_allowed[args.labeled_bs:], 
                                                             pos_weight=pos_weight)
            pseudo_supervision2 = dice_loss(out2_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1))
            # pseudo_supervision2_cls = F.binary_cross_entropy_with_logits(out2_cls[args.labeled_bs:], pseudo_outputs1_cls, reduction="none", pos_weight=pos_weight)
            pseudo_supervision2_cls = masked_bce_with_logits(out2_cls[args.labeled_bs:], 
                                                             pseudo_outputs1_cls.float(), 
                                                             mask_x_allowed[args.labeled_bs:], 
                                                             pos_weight=pos_weight)
            # print(pseudo_supervision1)
            
            # contra = ConstraLoss_AvgProj(outputs1, outputs2, ndf=1)
            target_pred1 = F.softmax(out1, dim=1)
            target_pred2 = F.softmax(out2, dim=1)
            # l_cdd1 = discrepancy_calc(target_pred1, target_pred2)
            # l_cdd = discrepancy_calc_2(target_pred1, target_pred2)

            model1_loss = loss1 + loss1_cls + consistency_weight_cps * 0.5 * (pseudo_supervision1 + pseudo_supervision1_cls)
            model2_loss = loss2 + loss2_cls + consistency_weight_cps * 0.5 * (pseudo_supervision2 + pseudo_supervision2_cls)

            # model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.5*consistency_loss
            # model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.5*consistency_loss

            # model1_loss = loss1 + consistency_weight * pseudo_supervision1 + 0.5*new_loss
            # model2_loss = loss2 + consistency_weight * pseudo_supervision2 + 0.5*new_loss

            loss = model1_loss + model2_loss #+ contra_weight_contra * l_cdd #+ consistency_weight_mt * consistency_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
            
            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight_cps, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (
                iter_num, model1_loss.item(), model2_loss.item()))
            
            if iter_num > 0 and iter_num % (labeled_slice / args.labeled_bs) == 0:
            # if iter_num > 0 and iter_num % 10 == 0:
                model1.eval()
                model2.eval()
                
                metric_dice1 = 0.0
                metric_dice2 = 0.0
                C = args.seg_num_classes
                K = args.cls_num_classes 
                tol = 2.0
                
                dice_sum_1 = np.zeros(C - 1, dtype=np.float64)
                dice_sum_2 = np.zeros(C - 1, dtype=np.float64)
                nsd_sum_1 = np.zeros(C - 1, dtype=np.float64)
                nsd_sum_2 = np.zeros(C - 1, dtype=np.float64)
                cnt_1 = np.zeros(C - 1, dtype=np.int64)
                cnt_2 = np.zeros(C - 1, dtype=np.int64)

                y_true_all, y_prob_1_all, y_prob_2_all, views_all = [], [], [], []                
                
                for i_batch, ((image_x, view_x, mask_x, class_label_x)) in enumerate(valloader):
                    image = image_x.to(device)
                    view = view_x.to(device).long().view(-1)
                    gt_mask = mask_x.to(device)
                    class_label = class_label_x.to(device)   
                    
                    h, w = image.shape[-2:]
                    image_rs = F.interpolate(
                        image, (args.resize_target, args.resize_target),
                        mode="bilinear", align_corners=False
                    )                    

                    # volume_valbatch, label_valbatch = sampled_batch['image'], sampled_batch['label']
                    # volume_valbatch, label_valbatch = volume_valbatch.to(device), label_valbatch.to(device)
                    # label_valbatch = torch.squeeze(label_valbatch,1)
               
                    preds_1, preds_cls_1 = model1(image_rs)
                    preds_2, preds_cls_2 = model2(image_rs)
                    preds_1_logits = F.interpolate(
                        preds_1, (h, w),
                        mode="bilinear", align_corners=False
                    )
                    preds_2_logits = F.interpolate(
                        preds_2, (h, w),
                        mode="bilinear", align_corners=False
                    )                    
                    
                    # preds_1_soft = torch.softmax(preds_1, dim=1)
                    # preds_2_soft = torch.softmax(preds_2, dim=1)
                    
                    pm_1 = preds_1_logits.argmax(dim=1)
                    pm_2 = preds_2_logits.argmax(dim=1)

                    pm_1 = pm_1.cpu().numpy()[0].astype(np.int32)
                    pm_2 = pm_2.cpu().numpy()[0].astype(np.int32)
                    gt = gt_mask.cpu().numpy()[0].astype(np.int32)  
                    
                    for cls in range(1, C):
                        pred_bin_1 = (pm_1 == cls)
                        pred_bin_2 = (pm_2 == cls)
                        gt_bin = (gt == cls)
                        union_1 = pred_bin_1.sum() + gt_bin.sum()
                        union_2 = pred_bin_1.sum() + gt_bin.sum()
                        
                        if union_1 != 0:
                            inter_1 = (pred_bin_1 & gt_bin).sum()
                            dice_sum_1[cls - 1] += (2.0 * inter_1) / (union_1 + 1e-8)
                            nsd_sum_1[cls - 1] += nsd_binary(pred_bin_1, gt_bin, tol=tol)
                            cnt_1[cls - 1] += 1 
                        
                        if union_2 != 0:
                            inter_2 = (pred_bin_2 & gt_bin).sum()
                            dice_sum_2[cls - 1] += (2.0 * inter_2) / (union_2 + 1e-8)
                            nsd_sum_2[cls - 1] += nsd_binary(pred_bin_2, gt_bin, tol=tol)
                            cnt_2[cls - 1] += 1     
                        
                    prob_1 = torch.sigmoid(preds_cls_1)
                    prob_2 = torch.sigmoid(preds_cls_2)
                    y_true_all.append(class_label.detach().cpu().numpy()[0])
                    y_prob_1_all.append(prob_1.detach().cpu().numpy()[0])
                    y_prob_2_all.append(prob_2.detach().cpu().numpy()[0])
                    views_all.append(view.detach().cpu().numpy()[0]) 
                    
                dice_class_1 = 100.0 * dice_sum_1 / np.maximum(cnt_1, 1)
                dice_class_2 = 100.0 * dice_sum_2 / np.maximum(cnt_2, 1)
                nsd_class_1 = 100.0 * nsd_sum_1 / np.maximum(cnt_1, 1)
                nsd_class_2 = 100.0 * nsd_sum_2 / np.maximum(cnt_2, 1)

                valid_mask_1 = cnt_1 > 0
                valid_mask_2 = cnt_2 > 0
                mean_dice_1 = float(dice_class_1[valid_mask_1].mean()) if valid_mask_1.any() else 0.0
                mean_dice_2 = float(dice_class_2[valid_mask_2].mean()) if valid_mask_2.any() else 0.0
                mean_nsd_1 = float(nsd_class_1[valid_mask_1].mean()) if valid_mask_1.any() else 0.0
                mean_nsd_2 = float(nsd_class_2[valid_mask_2].mean()) if valid_mask_2.any() else 0.0

                y_true_all = np.stack(y_true_all, axis=0) if len(y_true_all) else np.zeros((0, K), dtype=np.float32)
                y_prob_1_all = np.stack(y_prob_1_all, axis=0) if len(y_prob_1_all) else np.zeros((0, K), dtype=np.float32)
                y_prob_2_all = np.stack(y_prob_2_all, axis=0) if len(y_prob_2_all) else np.zeros((0, K), dtype=np.float32)
                views_all = np.array(views_all, dtype=np.int32) if len(views_all) else np.zeros((0,), dtype=np.int32)

                metrics_1 = masked_metrics_with_threshold_search(y_true_all, y_prob_1_all, views_all, cls_allowed)
                macro_f1_1 = float(metrics_1["macro_f1@0.5"])
                score_1 = (mean_dice_1 + mean_nsd_1) / 2.0 + macro_f1_1 * 100.0    
                
                metrics_2 = masked_metrics_with_threshold_search(y_true_all, y_prob_2_all, views_all, cls_allowed)
                macro_f1_2 = float(metrics_2["macro_f1@0.5"])
                score_2 = (mean_dice_2 + mean_nsd_2) / 2.0 + macro_f1_2 * 100.0                  
                
                writer.add_scalar('info/model1_score_1', score_1, iter_num)
                writer.add_scalar('info/model2_score_2', score_2, iter_num)
                
                logging.info(
                    # f"dice_class_1={dice_class_1}, "
                    # f"nsd_class_1={nsd_class_1}, "
                    f"mean_dice_1={mean_dice_1}, "
                    f"mean_nsd_1={mean_nsd_1}, "
                    f"macro_f1_1={macro_f1_1}, "
                    f"score_1={float(score_1)}"
                )  
                logging.info(
                    # f"dice_class_2={dice_class_2}, "
                    # f"nsd_class_2={nsd_class_2}, "
                    f"mean_dice_2={mean_dice_2}, "
                    f"mean_nsd_2={mean_nsd_2}, "
                    f"macro_f1_2={macro_f1_2}, "
                    f"score_2={float(score_2)}"
                )                         
                performance1 = score_1
                performance2 = score_2

                if performance1 > best_performance1 and iter_num > (max_iterations / 2):
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, 
                                                      round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)
                    
                if performance2 > best_performance2 and iter_num > (max_iterations / 2):
                    best_performance2 = performance2
                    # save_mode_path = os.path.join(snapshot_path,
                    #                               'model2_iter_{}_dice_{}.pth'.format(
                    #                                   iter_num, 
                    #                                   round(best_performance2, 4)))
                    # save_best = os.path.join(snapshot_path,'{}_best_model2.pth'.format(args.model))
                    # torch.save(model2.state_dict(), save_mode_path)
                    # torch.save(model2.state_dict(), save_best)              
                
                model1.train()
                model2.train()

            # if iter_num % 5000 == 0:
            if iter_num == max_iterations:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                # save_mode_path = os.path.join(
                #     snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                # torch.save(model2.state_dict(), save_mode_path)
                # logging.info("save model2 to {}".format(save_mode_path))
                
            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "{}/{}_{}".format(
        args.root_path, args.exp, args.labeled_num)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)    