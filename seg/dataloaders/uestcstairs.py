from seg.base import BaseDataSet, BaseDataLoader
from seg.utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class StairDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(StairDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'sData')
        self.image_dir = os.path.join(self.root, 'png')
        self.label_dir = os.path.join(self.root, 'Seg')

        file_list = os.path.join(self.root, "list",self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id)
        label_path = os.path.join(self.label_dir, image_id)
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_new = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image_new, label, image_id


class Stair(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        if split in ["train", "trainval", "val", "test"]:
            self.dataset = StairDataset(**kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super(Stair, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
