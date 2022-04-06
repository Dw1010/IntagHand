import argparse
import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import load_model
from models.manolayer import ManoLayer
from utils.config import load_cfg
from utils.vis_utils import mano_two_hands_renderer
from utils.utils import get_mano_path
from dataset.dataset_utils import IMG_SIZE, cut_img
from dataset.interhand import fix_shape, InterHand_dataset


class Jr():
    def __init__(self, J_regressor,
                 device='cuda'):
        self.device = device
        self.process_J_regressor(J_regressor)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0
        tip_regressor[1, 317] = 1.0
        tip_regressor[2, 444] = 1.0
        tip_regressor[3, 556] = 1.0
        tip_regressor[4, 673] = 1.0
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
        new_order = [0, 13, 14, 15, 16,
                     1, 2, 3, 17,
                     4, 5, 6, 18,
                     10, 11, 12, 19,
                     7, 8, 9, 20]
        self.J_regressor = J_regressor[new_order].contiguous().to(self.device)

    def __call__(self, v):
        return torch.matmul(self.J_regressor, v)


class handDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask, dense, hand_dict = self.dataset[idx]
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        maskTensor = torch.tensor(mask, dtype=torch.float32) / 255

        joints_left_gt = torch.from_numpy(hand_dict['left']['joints3d']).float()
        verts_left_gt = torch.from_numpy(hand_dict['left']['verts3d']).float()
        joints_right_gt = torch.from_numpy(hand_dict['right']['joints3d']).float()
        verts_right_gt = torch.from_numpy(hand_dict['right']['verts3d']).float()

        return imgTensor, maskTensor, joints_left_gt, verts_left_gt, joints_right_gt, verts_right_gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    parser.add_argument("--model", type=str, default='misc/model/interhand.pth')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--bs", type=int, default=32)
    opt = parser.parse_args()

    opt.map = False

    network = load_model(opt.cfg)

    state = torch.load(opt.model, map_location='cpu')
    try:
        network.load_state_dict(state)
    except:
        state2 = {}
        for k, v in state.items():
            state2[k[7:]] = v
        network.load_state_dict(state2)

    network.eval()
    network.cuda()

    mano_path = get_mano_path()
    mano_layer = {'left': ManoLayer(mano_path['left'], center_idx=None),
                  'right': ManoLayer(mano_path['right'], center_idx=None)}
    fix_shape(mano_layer)
    J_regressor = {'left': Jr(mano_layer['left'].J_regressor),
                   'right': Jr(mano_layer['right'].J_regressor)}

    faces_left = mano_layer['left'].get_faces()
    faces_right = mano_layer['right'].get_faces()

    dataset = handDataset(InterHand_dataset(opt.data_path, split='test'))
    dataloader = DataLoader(dataset, batch_size=opt.bs, shuffle=False,
                            num_workers=4, drop_last=False, pin_memory=True)

    joints_loss = {'left': [], 'right': []}
    verts_loss = {'left': [], 'right': []}

    with torch.no_grad():
        for data in tqdm(dataloader):

            imgTensors = data[0].cuda()
            joints_left_gt = data[2].cuda()
            verts_left_gt = data[3].cuda()
            joints_right_gt = data[4].cuda()
            verts_right_gt = data[5].cuda()

            joints_left_gt = J_regressor['left'](verts_left_gt)
            joints_right_gt = J_regressor['right'](verts_right_gt)

            root_left_gt = joints_left_gt[:, 9:10]
            root_right_gt = joints_right_gt[:, 9:10]
            length_left_gt = torch.linalg.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
            length_right_gt = torch.linalg.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
            joints_left_gt = joints_left_gt - root_left_gt
            verts_left_gt = verts_left_gt - root_left_gt
            joints_right_gt = joints_right_gt - root_right_gt
            verts_right_gt = verts_right_gt - root_right_gt

            result, paramsDict, handDictList, otherInfo = network(imgTensors)

            verts_left_pred = result['verts3d']['left']
            verts_right_pred = result['verts3d']['right']
            joints_left_pred = J_regressor['left'](verts_left_pred)
            joints_right_pred = J_regressor['right'](verts_right_pred)

            root_left_pred = joints_left_pred[:, 9:10]
            root_right_pred = joints_right_pred[:, 9:10]
            length_left_pred = torch.linalg.norm(joints_left_pred[:, 9] - joints_left_pred[:, 0], dim=-1)
            length_right_pred = torch.linalg.norm(joints_right_pred[:, 9] - joints_right_pred[:, 0], dim=-1)
            scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
            scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)

            joints_left_pred = (joints_left_pred - root_left_pred) * scale_left
            verts_left_pred = (verts_left_pred - root_left_pred) * scale_left
            joints_right_pred = (joints_right_pred - root_right_pred) * scale_right
            verts_right_pred = (verts_right_pred - root_right_pred) * scale_right

            joint_left_loss = torch.linalg.norm((joints_left_pred - joints_left_gt), ord=2, dim=-1)
            joint_left_loss = joint_left_loss.detach().cpu().numpy()
            joints_loss['left'].append(joint_left_loss)

            joint_right_loss = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
            joint_right_loss = joint_right_loss.detach().cpu().numpy()
            joints_loss['right'].append(joint_right_loss)

            vert_left_loss = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
            vert_left_loss = vert_left_loss.detach().cpu().numpy()
            verts_loss['left'].append(vert_left_loss)

            vert_right_loss = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
            vert_right_loss = vert_right_loss.detach().cpu().numpy()
            verts_loss['right'].append(vert_right_loss)

    joints_loss['left'] = np.concatenate(joints_loss['left'], axis=0)
    joints_loss['right'] = np.concatenate(joints_loss['right'], axis=0)
    verts_loss['left'] = np.concatenate(verts_loss['left'], axis=0)
    verts_loss['right'] = np.concatenate(verts_loss['right'], axis=0)

    joints_mean_loss_left = joints_loss['left'].mean() * 1000
    joints_mean_loss_right = joints_loss['right'].mean() * 1000
    verts_mean_loss_left = verts_loss['left'].mean() * 1000
    verts_mean_loss_right = verts_loss['right'].mean() * 1000

    print('joint mean error:')
    print('    left: {} mm, right: {} mm'.format(joints_mean_loss_left, joints_mean_loss_right))
    print('    all: {} mm'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
    print('vert mean error:')
    print('    left: {} mm, right: {} mm'.format(verts_mean_loss_left, verts_mean_loss_right))
    print('    all: {} mm'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))
