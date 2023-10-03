import os
import sys
import argparse
import logging
import random
from tqdm import tqdm
import torch
import gorilla
import pickle as pkl
import numpy as np
import psutil
from file_utils import get_open_fds

torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'sphericalmap_utils'))
sys.path.append(os.path.join(BASE_DIR, 'lib', 'pointnet2'))

from solver import test_func, get_logger
from dataset_dino import TestDataset, TrainingDataset
from evaluation_utils import evaluate

def get_parser():
    parser = argparse.ArgumentParser(
        description="VI-Net")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/base.yaml",
                        help="path to config file")
    parser.add_argument("--dataset",
                        type=str,
                        default="REAL275",
                        help="[REAL275 | CAMERA25]")
    parser.add_argument("--test_epoch",
                        type=int,
                        default=0,
                        help="test epoch")
    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    
    cfg = gorilla.Config.fromfile(args.config)
    cfg.mod = 'r'
    cfg.dataset = args.dataset
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.log_dir = os.path.join('log', args.dataset)
    cfg.save_path = os.path.join(cfg.log_dir, 'results')
    if not os.path.isdir(cfg.save_path):
        os.makedirs(cfg.save_path)
    

    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=cfg.log_dir+"/test_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)
    feature_path = os.path.join(BASE_DIR, cfg.feature.feature_path)
    if not os.path.isdir(feature_path):
        os.makedirs(feature_path)

    

    

    
    train_dataset  = TrainingDataset(
        cfg.train_dataset,
        cfg.dataset,
        cfg.mod,
        resolution = cfg.resolution,
        ds_rate = cfg.ds_rate,
        num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.train_dataloader.bs, 
        )
    # data loader
    train_dataloder = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.bs,
        num_workers=int(cfg.train_dataloader.num_workers),
        shuffle=False,
        sampler=None,
        drop_last=False,
        pin_memory=cfg.train_dataloader.pin_memory
    )
    
    from extractor_dino import ViTExtractor
    from torchvision import transforms
    extractor = ViTExtractor('dinov2_vits14', 14, device = 'cuda')
    extractor_preprocess = transforms.Normalize(mean=extractor.mean, std=extractor.std)
    def extract_feature(rgb_raw):
        rgb_raw = rgb_raw.permute(0,3,1,2)
        
        rgb_raw = extractor_preprocess(rgb_raw)
        
        
        with torch.no_grad():
        
            dino_feature = extractor.extract_descriptors(rgb_raw, layer = 11, facet = 'token' )
        
        dino_feature = dino_feature.reshape(dino_feature.shape[0],6720//14,6720//14,-1)
        
        return dino_feature.contiguous()
    
    with tqdm(total=len(train_dataloder)) as t:
        for i, data in enumerate(train_dataloder):
            
            with torch.no_grad():
                features = extract_feature(data['rgb']).cpu()
            import pdb;pdb.set_trace()


            num_instance = features.shape[0]
            #import pdb;pdb.set_trace()
            
            



            t.set_description(
                "Test [{}/{}][{}]: ".format(i+1, len(train_dataloder), num_instance)
            )
            
            t.update(1)
        
    
    
        

    
