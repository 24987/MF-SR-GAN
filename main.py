from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from FID import mainFID
import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        # default='cfg/bird_attn2.yml', type=str)
                        # default='cfg/eval_coco.yml', type=str)
                        # default='cfg/eval_bird.yml', type=str)
                        # default='cfg/eval_flower.yml', type=str)
                        default='cfg/flower_attn2.yml', type=str)
    			# default='cfg/coco_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize((int(imsize * 76 / 64), int(imsize * 76 / 64))),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    if cfg.TRAIN.FLAG:
        from datasets import TextDataset

        dataset = TextDataset(cfg.DATA_DIR, split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
        print(len(dataloader))
        algo = trainer(output_dir, dataloader, dataset.n_words)
        from datasets import prepare_data_fake_real
        algo.train_fake_real(prepare_data_fake_real)