import argparse
import datetime
import logging
import math
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import Datasets_seg
from utils.get_model import create_model

warnings.simplefilter("ignore")


############################## Config Loading ##############################
parser = argparse.ArgumentParser(description="Train Segmentation Model")
parser.add_argument(
    "--config", 
    type=str, 
    required=True, 
    help="Path to the config yaml file (e.g., config/train_seg/316L_Grains.yaml)"
)
args = parser.parse_args()

with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_name = config["model"]["name"]
model_args = config["model"]["args"]

dataset_name = config["dataset"]["name"]
csv_dir_path = config["dataset"]["csv_dir_path"]
input_col = config["dataset"]["input_col"]
target_col = config["dataset"]["target_col"]
train_data_aug = config["dataset"]["train_data_aug"]

epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
accumulation_steps = config["training"]["accumulation_steps"]
num_folds = config["training"]["num_folds"]
save_start_epoch = config["training"].get("save_start_epoch", 1)

optimizer_conf = config["optimizer"]
scheduler_conf = config["scheduler"]
############################################################################


def print_and_logging(message):
    print(message)
    logging.info(message)


base_output_dir = os.path.join("results_seg", dataset_name, model_name)
if os.path.exists(base_output_dir):
    version = 2
    while os.path.exists(f"{base_output_dir}_v{version}"):
        version += 1
    base_output_dir = f"{base_output_dir}_v{version}"
os.makedirs(base_output_dir)


####################################################################
# Cross-Validation Loop (Training only)
####################################################################
for fold in range(1, num_folds + 1):
    print(f"\n{'='*20} Starting Fold {fold} {'='*20}")

    ####################################################################
    # Set seed
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = False

    output_dir = os.path.join(base_output_dir, f"fold_{fold}")
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir_path, f"fold_{fold}_paths.csv")

    log_path = os.path.join(output_dir, f"{model_name}_fold{fold}.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", filename=log_path
    )

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    logging.info(now.strftime("%Y/%m/%d %H:%M\n"))
    logging.info(f"Dataset: {csv_dir_path}")
    logging.info(f"Fold: {fold}")
    logging.info(f"Model: {model_name}")
    logging.info(f"Model config: {model_args}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Batch size (per step): {batch_size}")
    logging.info(f"Accumulation steps: {accumulation_steps}")
    logging.info(f"Effective batch size: {batch_size * accumulation_steps}")
    logging.info(f"Optimizer config: {optimizer_conf}")
    logging.info(f"Scheduler config: {scheduler_conf}")

    ####################################################################
    # Create dataloaders
    train_dataset = Datasets_seg(
        csv_file=csv_file, 
        phase='train', 
        input_col=input_col, 
        target_col=target_col,
        data_aug=train_data_aug
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
    )

    print_and_logging(f"===> Number of training images: {len(train_dataset)}")

    ####################################################################
    # Init params
    num_iters = len(train_loader)
    num_update_steps_per_epoch = math.ceil(num_iters / accumulation_steps)
    total_update_steps = epochs * num_update_steps_per_epoch

    model = create_model(model_name, model_args)
    model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_conf["lr"],
        weight_decay=optimizer_conf["weight_decay"],
        betas=optimizer_conf["betas"],
    )
    
    if scheduler_conf["flg"]:
        warmup_flg = scheduler_conf.get("warmup_flg", False)
        warmup_steps = scheduler_conf.get("warmup_steps", 0) if warmup_flg else 0
        warmup_start_lr = scheduler_conf.get("warmup_start_lr", 0.0)
        
        main_steps = total_update_steps - warmup_steps
        
        if scheduler_conf["name"] == "cosine_annealing":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=main_steps,
                eta_min=scheduler_conf["eta_min"]
            )
        elif scheduler_conf["name"] == "poly":
            main_scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, 
                total_iters=main_steps,
                power=1.0
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_conf['name']}")
            
        if warmup_flg and warmup_steps > 0:
            start_factor = warmup_start_lr / optimizer_conf["lr"]
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=start_factor, 
                end_factor=1.0, 
                total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[warmup_steps]
            )
        else:
            scheduler = main_scheduler
            
    criterion = nn.BCEWithLogitsLoss()

    ####################################################################
    # Training
    print(f"===> Model: {model_name}")
    print(f"===> Total epochs: {epochs}, Effective Batch size: {batch_size * accumulation_steps}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        optimizer.zero_grad()

        for i, data in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"), 1
        ):
            gt = data[0].cuda()
            inp = data[1].cuda()

            out = model(inp)
            loss = criterion(out, gt)

            loss = loss / accumulation_steps
            loss.backward()

            if i % accumulation_steps == 0 or i == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler_conf["flg"]:
                    scheduler.step()
            
            epoch_loss += loss.item() * accumulation_steps

        epoch_loss /= num_iters

        current_lr = optimizer.param_groups[0]['lr']
        print_and_logging(
            "Epoch: {}   Time: {}m   Loss: {:.4f}   LR: {:.6f}".format(
                epoch, int((time.time() - epoch_start_time) // 60), epoch_loss, current_lr
            )
        )

        if epoch >= save_start_epoch:
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    print_and_logging(f"Fold {fold} Training Finished.")