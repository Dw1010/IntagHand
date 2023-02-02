import sys
import os
from tkinter.messagebox import NO
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import random
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer


from models.model import load_model

from utils.tb_utils import tbUtils
from utils.lr_sc import StepLR_withWarmUp
from utils.DataProvider import DataProvider
from utils.vis_utils import mano_two_hands_renderer
from utils.utils import get_mano_path

from core.loader import handDataset
from core.Loss import GraphLoss, calc_loss_GCN
from core.vis_train import tb_vis_train_gcn
from dataset.dataset_utils import IMG_SIZE, BLUR_KERNEL
from dataset.inference import get_final_preds2


def freeze_model(model):
    for (name, params) in model.named_parameters():
        params.requires_grad = False


def train_gcn(rank=0, world_size=1, cfg=None, dist_training=False):
    if dist_training:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(cfg.TRAIN.DIST_PORT)
        print("Init distributed training on local rank {}".format(rank))
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    mano_path = get_mano_path()

    # -------------------------------------------------
    # | 1. load model/optimizer/scheduler/tensorboard |
    # -------------------------------------------------
    # load network
    network = load_model(cfg)
    network.to(rank)

    if cfg.MODEL.freeze_upsample:
        freeze_model(network.decoder.unsample_layer)

    converter = {}
    for hand_type in ['left', 'right']:
        converter[hand_type] = network.decoder.converter[hand_type]

    if dist_training:
        network = DDP(
            network, device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
        )
    # print('local rank {}: init model, done'.format(rank))

    # load optimizer
    optim_params = list(filter(lambda p: p.requires_grad, network.parameters()))
    if cfg.TRAIN.OPTIM == 'adam':
        if dist_training:
            optimizer = ZeroRedundancyOptimizer(
                optim_params,
                optimizer_class=torch.optim.Adam,
                lr=cfg.TRAIN.LR
            )
        else:
            optimizer = torch.optim.Adam(optim_params, lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIM == 'rms':
        if dist_training:
            optimizer = ZeroRedundancyOptimizer(
                optim_params,
                optimizer_class=torch.optim.RMSprop,
                lr=cfg.TRAIN.LR
            )
        else:
            optimizer = torch.optim.RMSprop(optim_params, lr=cfg.TRAIN.LR)
    else:
        raise ValueError('wrong optimizer type')
    # print('local rank {}: init optimizer, done'.format(rank))

    # load learning rate scheduler
    lr_scheduler = StepLR_withWarmUp(optimizer,
                                     last_epoch=-1 if cfg.TRAIN.current_epoch == 0 else cfg.TRAIN.current_epoch,
                                     init_lr=1e-3 * cfg.TRAIN.LR,
                                     warm_up_epoch=cfg.TRAIN.warm_up,
                                     gamma=cfg.TRAIN.lr_decay_gamma,
                                     step_size=cfg.TRAIN.lr_decay_step,
                                     min_thres=0.05)
    # print('local rank {}: init lr_scheduler, done'.format(rank))

    if rank == 0:
        # tensorboard
        writer = SummaryWriter(cfg.TB.SAVE_DIR)
        renderer = mano_two_hands_renderer(img_size=IMG_SIZE, device='cuda:{}'.format(rank))

    # --------------------------
    # | 2. load dataset & Loss |
    # --------------------------
    aux_lambda = 2**(6 - len(cfg.MODEL.DECONV_DIMS))
    trainDataset = handDataset(mano_path=mano_path,
                               interPath=cfg.DATASET.INTERHAND_PATH,
                               theta=[-cfg.DATA_AUGMENT.THETA, cfg.DATA_AUGMENT.THETA],
                               scale=[1 - cfg.DATA_AUGMENT.SCALE, 1 + cfg.DATA_AUGMENT.SCALE],
                               uv=[-cfg.DATA_AUGMENT.UV, cfg.DATA_AUGMENT.UV],
                               aux_size=IMG_SIZE // aux_lambda)
    # print('local rank {}: init dataset, done'.format(rank))

    provider_train = DataProvider(dataset=trainDataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=4, dist=dist_training)
    train_batch_per_epoch = provider_train.batch_per_epoch
    # print('local rank {}: init data loader, done'.format(rank))

    Loss = {}
    faces = {}
    for hand_type in ['left', 'right']:
        with open(mano_path[hand_type], 'rb') as file:
            manoData = pickle.load(file, encoding='latin1')
        J_regressor = manoData['J_regressor'].tocoo(copy=False)
        location = []
        data = []
        for i in range(J_regressor.data.shape[0]):
            location.append([J_regressor.row[i], J_regressor.col[i]])
            data.append(J_regressor.data[i])
        i = torch.LongTensor(location)
        v = torch.FloatTensor(data)
        J_regressor = torch.sparse.FloatTensor(i.t(), v, torch.Size([16, 778])).to_dense()
        Loss[hand_type] = GraphLoss(J_regressor, manoData['f'],
                                    level=4,
                                    device=rank)
        # device='cuda:{}'.format(rank))
        faces[hand_type] = manoData['f']

    # print('local rank {}: init training loss, done'.format(rank))

    # ------------
    # | 3. train |
    # ------------
    # print('local rank {}: strat training'.format(rank))
    for epoch in range(cfg.TRAIN.current_epoch, cfg.TRAIN.EPOCHS):
        network.train()
        train_bar = range(train_batch_per_epoch)
        if rank == 0:
            train_bar = tqdm(train_bar)
        for bIdx in train_bar:
            total_idx = epoch * train_batch_per_epoch + bIdx

            # ------------
            # | training |
            # ------------
            label_list = provider_train.next()
            label_list_out = []
            for label in label_list:
                if label is not None:
                    label_list_out.append(label.to(rank))
            [ori_img,
             imgTensors, mask, dense, hms,
             v2d_l, j2d_l, v2d_r, j2d_r,
             v3d_l, j3d_l, v3d_r, j3d_r,
             root_rel] = label_list_out
            result, paramsDict, handDictList, otherInfo = network(imgTensors)

            if cfg.MODEL.freeze_upsample:
                upsample_weight = None
            else:
                if dist_training:
                    upsample_weight = network.module.decoder.get_upsample_weight()
                else:
                    upsample_weight = network.decoder.get_upsample_weight()

            loss, aux_lost_dict, mano_loss_dict, coarsen_loss_dict = \
                calc_loss_GCN(cfg, epoch,
                              Loss['left'], Loss['right'],
                              converter['left'], converter['right'],
                              result, paramsDict, handDictList, otherInfo,
                              mask, dense, hms,
                              v2d_l, j2d_l, v2d_r, j2d_r,
                              v3d_l, j3d_l, v3d_r, j3d_r,
                              root_rel, img_size=imgTensors.shape[-1],
                              upsample_weight=upsample_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---------------
            # | tensorboard |
            # ---------------
            if rank == 0:
                writer.add_scalar('learning_rate', lr_scheduler.get_lr()[0], total_idx)
                writer.add_scalar('train/total_loss', loss.item(), total_idx)
                for k, v in mano_loss_dict.items():
                    if k != 'total_loss':
                        writer.add_scalar('train/mano_{}'.format(k), v.item(), total_idx)
                for k, v in aux_lost_dict.items():
                    if k != 'total_loss':
                        writer.add_scalar('train/aux_{}'.format(k), v.item(), total_idx)
                for k, v in coarsen_loss_dict.items():
                    if k != 'total_loss':
                        for t in range(len(v)):
                            writer.add_scalar('train/coarsen_{}_{}'.format(k, t), v[t].item(), total_idx)
                if (total_idx + 1) % cfg.TB.SHOW_GAP == 0:
                    tb_vis_train_gcn(cfg, writer, total_idx, renderer, v2d_l, v2d_r,
                                     ori_img, mask, dense,
                                     result, paramsDict, handDictList, otherInfo)

                    tbUtils.draw_MANO_joints(writer, 'hms/l_gt', total_idx, ori_img[0], j2d_l[0])
                    handJ2d_pred, _ = get_final_preds2(otherInfo['hms'][:, :21].detach().cpu().numpy(), BLUR_KERNEL)
                    handJ2d_pred = torch.from_numpy(handJ2d_pred) * aux_lambda
                    tbUtils.draw_MANO_joints(writer, 'hms/l_pred', total_idx, ori_img[0], handJ2d_pred[0])

                    tbUtils.draw_MANO_joints(writer, 'hms/r_gt', total_idx, ori_img[0], j2d_r[0])
                    handJ2d_pred, _ = get_final_preds2(otherInfo['hms'][:, 21:].detach().cpu().numpy(), BLUR_KERNEL)
                    handJ2d_pred = torch.from_numpy(handJ2d_pred) * aux_lambda
                    tbUtils.draw_MANO_joints(writer, 'hms/r_pred', total_idx, ori_img[0], handJ2d_pred[0])

                # --------
                # | tqdm |
                # --------
                train_bar.set_description('train, epoch:{}'.format(epoch))
                train_bar.set_postfix(totalLoss=loss.item())

        lr_scheduler.step()
        if (epoch + 1) % cfg.SAVE.SAVE_GAP == 0:
            if rank == 0:  # save checkpoint in main process
                torch.save(network.state_dict(), os.path.join(cfg.SAVE.SAVE_DIR, str(epoch + 1) + '.pth'))

    if dist_training:
        dist.barrier()
        dist.destroy_process_group()
