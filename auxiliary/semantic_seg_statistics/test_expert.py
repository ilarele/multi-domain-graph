import os
import shutil
import sys
import numpy as np
import torch
import cv2
import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import experts.semantic_segmentation_expert

NYU_classes = [
    'none', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk',
    'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floormat', 'clothes',
    'ceiling', 'books', 'refrigerator', 'television', 'paper', 'towel',
    'showercurtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet',
    'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture',
    'otherprop'
]

classes_of_NYU_covered_by_ADE = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 17, 22])

ADE_2_NYU = np.array([1, 22, 2, 8, 9, 3, 6, -1, 7, 4, 11, 5])

NYU_2_ADE = np.array([
    0, 1, 3, 6, 10, 12, 7, 9, 4, 5, 0, 11, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 2, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

ADE_classes = [
    'none', 'wall', 'ceiling', 'floor', 'door', 'window', 'cabinet', 'sofa',
    'railing', 'table', 'bed', 'painting', 'chair'
]
#0  0 [                          wall] occ 46.14% (din 100%: 46.14%)
#1   5 [                       ceiling] occ 16.45% (din 100%: 62.59%)
#2   3 [                floor;flooring] occ 10.95% (din 100%: 73.55%)
#3  14 [door;double;door+screen;door;screen] occ 5.00% (din 100%: 78.54%)
#4   8 [             windowpane;window] occ 3.48% (din 100%: 82.02%)
#5  10 [cabinet+wardrobe;closet;press+chest;of;drawers;chest;bureau;dresser] occ 3.51% (din 100%: 85.54%)
#6  23 [             sofa;couch;lounge] occ 1.59% (din 100%: 87.13%)
#7  38 [railing;rail+escalator;moving;staircase;moving;stairway+stairway;staircase+stairs;steps] occ 1.87% (din 100%: 89.00%)
#8  15 [table+coffee;table;cocktail;table] occ 1.38% (din 100%: 90.38%)
#9   7 [                           bed] occ 1.03% (din 100%: 91.41%)
#10  22 [              painting;picture] occ 0.96% (din 100%: 92.37%)
#11  19 [chair+armchair+swivel;chair+seat] occ 1.52% (din 100%: 93.88%)

# NYU_ourADE20K
# 1	 wall           - 1 - 0 [                          wall]
# 2	 floor          - 3 - 3 [                floor;flooring]
# 3	 cabinet        - 6 - 10 [cabinet+wardrobe;closet;press+chest;of;drawers;chest;bureau;dresser]
# 4	 bed            - 10 - 7 [                           bed]
# 5	 chair          - 12 - 19 [chair+armchair+swivel;chair+seat]
# 6	 sofa           - 7 - 23 [             sofa;couch;lounge]
# 7	 table          - 9 - 15 [table+coffee;table;cocktail;table]
# 8	 door           - 4 - 14 [door;double;door+screen;door;screen]
# 9	 window         - 5 - 8 [             windowpane;window]
# 10 bookshelf      -
# 11 picture        - 11 - 22 [              painting;picture]
# 12 counter        -
# 13 blinds         -
# 14 desk           -
# 15 shelves        -
# 16 curtain        -
# 17 dresser        - 6 - 10 [cabinet+wardrobe;closet;press+chest;of;drawers;chest;bureau;dresser]
# 18 pillow         -
# 19 mirror         -
# 20 floormat       - ---
# 21 clothes        -
# 22 ceiling        -  2 - 5 [                       ceiling]
# 23 books          -
# 24 refrigerator   -
# 25 television     -
# 26 paper          - ---
# 27 towel          -
# 28 showercurtain  - ---
# 29 box            -
# 30 whiteboard     - ---
# 31 person         -
# 32 nightstand     - ---
# 33 toilet         -
# 34 sink           -
# 35 lamp           -
# 36 bathtub        -
# 37 bag            -
# 38 otherstructure -
# 39 otherfurniture -
# 40 otherprop      -

train1_gt_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim_v2/train1/sem_seg'
train1_exp_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim_v2/train1/sem_seg_hrnet'
train1_rgb_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim_v2/train1/rgb'

train2_gt_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim_v2/train2/sem_seg'
train2_exp_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim_v2/train2/sem_seg_hrnet'
train2_rgb_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim_v2/train2/rgb'

test_gt_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim_v2/test/sem_seg'
test_exp_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim_v2/test/sem_seg_hrnet'
test_rgb_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim_v2/test/rgb'

valid_gt_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_gt/hypersim_v2/valid/sem_seg'
valid_exp_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim_v2/valid/sem_seg_hrnet'
valid_rgb_path = r'/data/multi-domain-graph-5/datasets/datasets_preproc_exp/hypersim_v2/valid/rgb'


def get_glob_paths(orig_path):
    glob_paths = []
    for orig_path_ in orig_path:
        glob_paths += ['%s/*.npy' % (orig_path_)]
    return glob_paths


def get_all_paths(glob_paths):
    all_paths = sorted(glob.glob(glob_paths[0]))
    for i in np.arange(1, len(glob_paths)):
        all_paths += sorted(glob.glob(glob_paths[i]))
    return all_paths


class SemanticSegDataset(Dataset):
    def __init__(self, rgb_dataset_path, gt_dataset_path, exp_dataset_path):
        super(SemanticSegDataset, self).__init__()

        self.rgb_paths = get_all_paths(get_glob_paths(rgb_dataset_path))
        self.gt_paths = get_all_paths(get_glob_paths(gt_dataset_path))
        self.exp_paths = get_all_paths(get_glob_paths(exp_dataset_path))

        print(len(self.rgb_paths))
        assert len(self.rgb_paths) == len(self.exp_paths)
        assert len(self.exp_paths) == len(self.gt_paths)

    def __getitem__(self, index):
        rgb = np.load(self.rgb_paths[index])
        exp = np.load(self.exp_paths[index])
        gt = np.load(self.gt_paths[index])
        return rgb, gt, exp

    def __len__(self):
        return len(self.gt_paths)


def get_iou(m1, m2):
    s_inter = torch.sum(m1[:, 0, :, :] * m2[:, 0, :, :], (1, 2))
    s_1 = torch.sum(m1[:, 0, :, :], (1, 2))
    s_2 = torch.sum(m2[:, 0, :, :], (1, 2))
    d = s_1 + s_2 - s_inter
    d[d == 0] = 1
    iou = s_inter / d  #(s_1 + s_2 - s_inter)
    return iou


def process_split(split_name, rgb_path, gt_path, exp_path):

    seg_expert = experts.semantic_segmentation_expert.SSegHRNet('hypersim')

    sem_seg_dataset = SemanticSegDataset(rgb_path, gt_path, exp_path)
    dataloader = DataLoader(sem_seg_dataset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=20)

    n_all_pixels = 0
    all_classes = np.zeros((41))
    ious_per_cls = np.zeros(13)
    new_ious_per_cls = np.zeros(13)
    n_samples = 0
    for batch in tqdm(dataloader):
        rgb, gt, exp = batch

        comp_exp = seg_expert.apply_expert_batch(rgb)
        comp_exp = torch.tensor(comp_exp)

        n_all_pixels += torch.numel(gt)
        for cls_idx in range(1, 41):
            all_classes[cls_idx] += torch.sum(gt == cls_idx)
        all_classes[0] += torch.sum(gt < 1)

        trans_gt = gt.clone()
        trans_gt = trans_gt.long()
        transf = torch.tensor(NYU_2_ADE)
        trans_gt = transf[trans_gt]

        for i in range(0, 10):

            rgb_i = np.uint8(rgb[i, 0, :, :] * 255)
            gt_i = np.uint8((gt[i, 0, :, :] / 40) * 255)
            trans_gt_i = np.uint8((trans_gt[i, 0, :, :] / 12) * 255)
            exp_i = np.uint8(((exp[i, 0, :, :] + 1) / 12) * 255)
            comp_exp_i = np.uint8((comp_exp[i, 0, :, :] / 12) * 255)
            img0 = np.concatenate((rgb_i, gt_i, trans_gt_i, exp_i, comp_exp_i),
                                  1)
            cv2.imwrite('_test_%d.png' % i, np.uint8(img0))

        n_samples += rgb.shape[0]
        for ade_cls_idx in np.arange(1, 13):
            gt_mask = trans_gt == ade_cls_idx
            exp_mask = exp == ade_cls_idx - 1
            iou = get_iou(gt_mask, exp_mask)
            ious_per_cls[ade_cls_idx] += torch.sum(iou)
            new_exp_mask = comp_exp == ade_cls_idx
            new_iou = get_iou(gt_mask, new_exp_mask)
            new_ious_per_cls[ade_cls_idx] += torch.sum(new_iou)

    ious_per_cls = 100 * ious_per_cls / n_samples
    new_ious_per_cls = 100 * new_ious_per_cls / n_samples
    percents = 100 * all_classes / n_all_pixels

    print('----- %s -----' % split_name)
    for cls_idx in range(0, 41):
        print('%15s -- %4.2f' % (NYU_classes[cls_idx], percents[cls_idx]))

    expected_covered_by_expert = np.sum(
        percents[classes_of_NYU_covered_by_ADE])
    print('Expected to be covered by the expert -- %4.2f' %
          (expected_covered_by_expert))
    for cls_idx in np.arange(1, 13):
        print('%15s -- %4.2f -- %4.2f' %
              (ADE_classes[cls_idx], ious_per_cls[cls_idx],
               new_ious_per_cls[cls_idx]))


process_split(
    'all', [train1_rgb_path, train2_rgb_path, valid_rgb_path, test_rgb_path],
    [train1_gt_path, train2_gt_path, valid_gt_path, test_gt_path],
    [train1_exp_path, train2_exp_path, valid_exp_path, test_exp_path])
process_split('train1', [train1_rgb_path], [train1_gt_path], [train1_exp_path])
process_split('train2', [train2_rgb_path], [train2_gt_path], [train2_exp_path])
process_split('valid', [valid_rgb_path], [valid_gt_path], [valid_exp_path])
process_split('test', [test_rgb_path], [test_gt_path], [test_exp_path])
