import argparse
import logging
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from evaluate import evaluate
from model import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

train_root_dir = Path(r'')
val_root_dir = Path(r'')
dir_checkpoint = Path('./checkpoints/')
output_dir = Path('./output/')

def setup_logging():
    """Set up the logging format and level."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_dataloaders(img_scale, batch_size):
    """Create DataLoaders for training and validation."""
    train_dataset = BasicDataset(train_root_dir / 'images', train_root_dir / 'masks', img_scale)
    val_dataset = BasicDataset(val_root_dir / 'images', val_root_dir / 'masks', img_scale)
    
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
    
    return train_loader, val_loader

def get_optimizer(model, learning_rate, weight_decay, momentum):
    """Configure the optimizer."""
    return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def save_checkpoint_and_masks(model, dir_checkpoint, dataset_name, best_val_score, val_loader, device, amp):
    """Save the best model checkpoint and corresponding predicted masks."""
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    checkpoint_filename = f'{model.__class__.__name__}_{dataset_name}_best_[22222]_1217.pth'
    checkpoint_path = dir_checkpoint / checkpoint_filename
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f'Best model checkpoint saved as: {checkpoint_path}')
    
    # Save predicted masks
    output_subdir = output_dir / f'{model.__class__.__name__}_{dataset_name}_best_[22222]_1217'
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    model.eval()    
    with torch.no_grad():
        for batch in val_loader:
            images, true_masks, names = batch['image'], batch['mask'], batch['name']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_pred = model(images)
            masks_pred = torch.sigmoid(masks_pred) if model.n_classes == 1 else torch.softmax(masks_pred, dim=1)
            for i in range(images.size(0)):
                save_image(masks_pred[i], output_subdir / f'{names[i]}.png')
    
    logging.info(f'Predicted masks saved in: {output_subdir}')
    return best_val_score

class FocalLoss(nn.Module):
    """Focal Loss for binary and multi-class classification."""
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            # Reshape to (N, C, H, W) -> (N, C, H*W)
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            # Reshape to (N, C, H*W) -> (N, C*H*W)
            inputs = inputs.permute(0, 2, 1).contiguous().view(-1, inputs.size(1))
        
        targets = targets.view(-1)
        
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            inputs = inputs[valid]
            targets = targets[valid]
        
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.unsqueeze(1))
        logpt = logpt.squeeze(1)
        pt = logpt.exp()
        
        if len(inputs.shape) > 1 and inputs.size(1) > 1:
            loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        else:
            loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def get_criterion(model):
    """Define the loss function."""
    if model.n_classes > 1:
        return FocalLoss(alpha=1, gamma=2, reduction='mean')
    else:
        return FocalLoss(alpha=1, gamma=2, reduction='mean')

def train_model(
        model,
        device,
        epochs: int =300,
        batch_size: int =32,
        learning_rate: float = 1e-6,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-4,
        momentum: float = 0.99,
        gradient_clipping: float = 1.0,
):
    setup_logging()
    train_loader, val_loader = create_dataloaders(img_scale, batch_size)
    optimizer = get_optimizer(model, learning_rate, weight_decay, momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = get_criterion(model)
    global_step = 0
    best_val_score = 0  # Initialize the best validation score
    dataset_name = train_root_dir.parts[-1]

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss_focal = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss_dice = dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss_focal = criterion(masks_pred, true_masks)
                        loss_dice = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    loss = loss_focal + loss_dice

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                global_step += 1
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'loss (batch)': loss.item()})

        # Validation at the end of the epoch
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        logging.info(f'Validation Dice score: {val_score:.4f}')

        if save_checkpoint and val_score > best_val_score:
            best_val_score = save_checkpoint_and_masks(model, dir_checkpoint, dataset_name, val_score, val_loader, device, amp)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear).to(device)
    #model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! Enabling checkpointing to reduce memory usage, but this slows down training. Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        if hasattr(model, 'use_checkpointing'):
            model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
