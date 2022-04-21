
from data_utils.StairDataLoader import StairDataset
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--log_dir', type=str,default='2022-04-21_11-00', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')


    return parser.parse_args()


def test(model, loader):
    mean_wide_correct = []
    mean_height_correct = []
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        points=points.to(torch.float32)
        pred, _ = classifier(points)
        correct = (pred - target).abs()
        correct_all = (correct / target).sum(dim=0)
        wide_data = correct_all.data[0].item()
        height_data = correct_all.data[1].item()
        mean_wide_correct.append(wide_data / float(points.size()[0]))
        mean_height_correct.append(height_data / float(points.size()[0]))
    instance_wide_acc = np.mean(mean_wide_correct)
    instance_height_acc = np.mean(mean_height_correct)
    return 1-instance_wide_acc,1-instance_height_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/stairtest/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '/home/lzx/stair/point/data/'

    test_dataset = StairDataset(mode='test', num_pt=args.num_point, root=data_path, type=type,uniform=False,
                                process_data=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    '''MODEL LOADING'''

    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model( normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_wide_acc, instance_height_acc = test(classifier.eval(), testDataLoader)
        log_string('Test Wide Accuracy: %f,Test Height Accuracy: %f' % (instance_wide_acc, instance_height_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
