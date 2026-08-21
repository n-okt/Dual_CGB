import os
import glob
import torch
import random
import pickle
import pandas as pd
import numpy as np
from PIL import Image

class Datasets_seg(torch.utils.data.Dataset):
    def __init__(self, csv_file, phase='train', input_col='photo_path', target_col='boundary_path', data_aug=False):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['phase'] == phase].reset_index(drop=True)
        self.phase = phase
        self.input_col = input_col
        self.target_col = target_col
        self.data_aug = data_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        target_path = row[self.target_col]
        input_path = row[self.input_col]

        gt = np.asarray(Image.open(target_path).convert('L'))
        gt = np.where(gt > 127, 1.0, 0.0).astype(np.float32)
        gt = torch.from_numpy(gt).unsqueeze(0)  # Shape: (1, H, W)

        inp = np.asarray(Image.open(input_path).convert('RGB'))
        inp = inp.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1)  # Shape: (3, H, W)
        
        if self.phase == 'train' and self.data_aug:
            if random.random() > 0.5:
                inp = torch.flip(inp, dims=[2])
                gt = torch.flip(gt, dims=[2])
            
            if random.random() > 0.5:
                inp = torch.flip(inp, dims=[1])
                gt = torch.flip(gt, dims=[1])
            
            k = random.randint(0, 3)
            if k > 0:
                inp = torch.rot90(inp, k, dims=[1, 2])
                gt = torch.rot90(gt, k, dims=[1, 2])

        file_name = os.path.splitext(os.path.basename(target_path))[0]

        return gt, inp, file_name


class Datasets_res(torch.utils.data.Dataset):
    def __init__(self, csv_file, finetune, topo_dir="topo", phase="train", input_col="boundary_broken_path", target_col="boundary_path"):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['phase'] == phase].reset_index(drop=True)
        self.phase = phase
        self.input_col = input_col
        self.target_col = target_col
        self.topo_dir = topo_dir
        self.finetune = finetune

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        target_path = row[self.target_col]
        input_path = row[self.input_col]

        gt = np.asarray(Image.open(target_path).convert('L'))
        gt = np.where(gt > 127, 1.0, 0.0).astype(np.float32)
        gt = torch.from_numpy(gt).unsqueeze(0)

        inp = np.asarray(Image.open(input_path).convert('L'))
        inp = inp.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).unsqueeze(0)

        if self.finetune:
            # load topology data
            base_name = os.path.splitext(os.path.basename(target_path))[0]
            gt_dir = os.path.dirname(target_path)
            parent_dir = os.path.dirname(gt_dir)
            topo_path = os.path.join(parent_dir, self.topo_dir, f"{base_name}.pkl")
            with open(topo_path, 'rb') as f:
                gt_topo_data = pickle.load(f)

        if self.phase == 'test':
            if 'base_name' in row:
                file_name = f"{row['base_name']}_binary"
            else:
                file_name = f"{os.path.splitext(os.path.basename(target_path))[0]}_binary"
        else:
            file_name = os.path.splitext(os.path.basename(target_path))[0]

        if self.finetune:
            return gt, inp, gt_topo_data, file_name
        else:
            return gt, inp, None, file_name