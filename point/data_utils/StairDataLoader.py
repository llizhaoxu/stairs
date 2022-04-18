import pickle

import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import open3d as o3d
from tqdm import tqdm
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import cv2


class StairDataset(data.Dataset):
    def __init__(self, mode, num_pt, root,type, process_data=False):
        if mode == 'train':
            self.seg_path = '/root/data/point/data/train_data_seg.txt'
            self.depth_path = '/root/data/point/data/train_data_depth.txt'
        if mode == 'test':
            self.seg_path = '/root/data/point/data/test_data_seg.txt'
            self.depth_path = '/root/data/point/data/test_data_depth.txt'
        self.num_pt = num_pt
        self.root = root
        self.type=type
        self.process_data = process_data
        self.seg_list = []
        self.depth_list = []
        self.list_code = []

        self.cam_cx = 655.9583129882812
        self.cam_cy = 352.8207702636719
        self.cam_fx = 916.896240234375
        self.cam_fy = 917.2578735351562
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.front_num = 2
        input_file = open(self.seg_path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.seg_list.append(input_line)
        input_file.close()
        input_file = open(self.depth_path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.depth_list.append(input_line)
        input_file.close()
        self.length = len(self.seg_list)
        self.save_path = os.path.join(root, 'modelnet_%s_%dpts.dat' % (mode, self.num_pt))
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.depth_list)
                self.list_of_labels = [None] * len(self.depth_list)

                for index in tqdm(range(len(self.depth_list)), total=len(self.depth_list)):
                    depth = self.depth_list[index]
                    seg=self.seg_list[index]
                    truth=get_truth(depth,type)
                    cls = np.array(truth).astype(np.float32)
                    point_set = get_cloud(self.root,depth,seg,self.num_pt)
                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __getitem__(self, index):
        if self.process_data:
            point_set, truth = self.list_of_points[index], self.list_of_labels[index]
        else:
            imid=self.depth_list[index]
            if self.type=='wide':
                if imid[3]=='1':
                    truth = [0.35,0.123]
                if imid[3]=='2':
                    truth = [0.354,0.157]
                if imid[3]=='3':
                    truth = [0.354,0.157]
                if imid[3]=='4':
                    truth = [0.363,0.145]
                if imid[3]=='5':
                    truth = [0.305,0.145]
                if imid[3]=='6':
                    truth = [0.306,0.148]
                if imid[3]=='7':
                    truth = [0.325,0.153]
                if imid[3]=='8':
                    truth = [0.35,0.161]
                if imid[3]=='9':
                    truth = [0.318,0.155]
            else:
                if imid[3]=='1':
                    truth = [0.123]
                if imid[3]=='2':
                    truth = [0.157]
                if imid[3]=='3':
                    truth = [0.157]
                if imid[3]=='4':
                    truth = [0.145]
                if imid[3]=='5':
                    truth = [0.145]
                if imid[3]=='6':
                    truth = [0.148]
                if imid[3]=='7':
                    truth = [0.153]
                if imid[3]=='8':
                    truth = [0.155]
                if imid[3] == '9':
                    truth = [0.354]
            truth = torch.tensor(truth).float()
            cam_scale = 0.0010000000474974513
            depth = Image.open('{0}/depth/{1}'.format(self.root, self.depth_list[index]))
            depth = depth.resize((640, 480), Image.ANTIALIAS)
            depth = np.array(depth)
            mask = Image.open('{0}/segimg/{1}'.format(self.root, self.seg_list[index]))
            mask = mask.resize((640, 480), Image.ANTIALIAS)
            mask = np.array(mask)
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(mask, 1))
            mask = mask_label * mask_depth
            rmin, rmax, cmin, cmax = get_bbox(mask_label)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            cam_cx = self.cam_cx
            cam_cy = self.cam_cy
            cam_fx = self.cam_fx
            cam_fy = self.cam_fy

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

            pt2 = depth_masked * cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            clouds = np.concatenate((pt0, pt1, pt2), axis=1)
           # 点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(clouds)

            downpcd = pcd.voxel_down_sample(voxel_size=0.002)#下采样
            pcd_new, da = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)#去除离散值

            cloud_new = np.asarray(pcd_new.points)
            np.random.shuffle(cloud_new)
            point_set = cloud_new[0:self.num_pt, :]

        return point_set,truth

    def __len__(self):
        return self.length


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640
def get_truth(imid,type):
    if type == 'wide':
        if imid[3] == '1':
            truth = [0.35, 0.123]
        if imid[3] == '2':
            truth = [0.354, 0.157]
        if imid[3] == '3':
            truth = [0.354, 0.157]
        if imid[3] == '4':
            truth = [0.363, 0.145]
        if imid[3] == '5':
            truth = [0.305, 0.145]
        if imid[3] == '6':
            truth = [0.306, 0.148]
        if imid[3] == '7':
            truth = [0.325, 0.153]
        if imid[3] == '8':
            truth = [0.35, 0.161]
        if imid[3] == '9':
            truth = [0.318, 0.155]
    else:
        if imid[3] == '1':
            truth = [0.123]
        if imid[3] == '2':
            truth = [0.157]
        if imid[3] == '3':
            truth = [0.157]
        if imid[3] == '4':
            truth = [0.145]
        if imid[3] == '5':
            truth = [0.145]
        if imid[3] == '6':
            truth = [0.148]
        if imid[3] == '7':
            truth = [0.153]
        if imid[3] == '8':
            truth = [0.155]
        if imid[3] == '9':
            truth = [0.354]
    return truth
def get_cloud(root,depth,seg,num_pt):
    cam_scale = 0.0010000000474974513
    depth = Image.open('{0}/depth/{1}'.format(root, depth))
    depth = depth.resize((640, 480), Image.ANTIALIAS)
    depth = np.array(depth)
    mask = Image.open('{0}/segimg/{1}'.format(root, seg))
    mask = mask.resize((640, 480), Image.ANTIALIAS)
    mask = np.array(mask)
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(mask, 1))
    mask = mask_label * mask_depth
    rmin, rmax, cmin, cmax = get_bbox(mask_label)
    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

    cam_cx = 655.9583129882812
    cam_cy = 352.8207702636719
    cam_fx = 916.896240234375
    cam_fy = 917.2578735351562
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = depth_masked * cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    clouds = np.concatenate((pt0, pt1, pt2), axis=1)
    # 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(clouds)

    downpcd = pcd.voxel_down_sample(voxel_size=0.002)  # 下采样
    pcd_new, da = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)  # 去除离散值

    cloud_new = np.asarray(pcd_new.points)
    np.random.shuffle(cloud_new)
    point_set = cloud_new[0:num_pt, :]
    return  point_set

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax
