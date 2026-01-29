import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import custom modules (assumes they're in same directory)
from bss_dataset import create_dataloaders
from bss_model import ImprovedBSS, SemiSupervisedBSSLoss


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, accumulation_steps=4):
    """Train for one epoch with gradient accumulation"""
    model.train()
    
    total_loss = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        mix_mag = batch['mix_mag'].to(device)
        s1_mag = batch['s1_mag'].to(device)
        s2_mag = batch['s2_mag'].to(device)
        is_supervised = batch['is_supervised'].to(device)
        
        # Forward pass
        pred_s1, pred_s2, masks = model(mix_mag)
        
        # Compute loss
        loss_dict = criterion(pred_s1, pred_s2, s1_mag, s2_mag, mix_mag, is_supervised)
        loss = loss_dict['total'] / accumulation_steps  # Scale loss
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss_dict['total']
        total_sup_loss += loss_dict['supervised']
        total_unsup_loss += loss_dict['unsupervised'].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'sup': f"{loss_dict['supervised']:.4f}",
            'unsup': f"{loss_dict['unsupervised'].item():.4f}",
        })
    
    return {
        'loss': total_loss / n_batches,
        'supervised_loss': total_sup_loss / n_batches,
        'unsupervised_loss': total_unsup_loss / n_batches,
    }


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    
    total_loss = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device
            mix_mag = batch['mix_mag'].to(device)
            s1_mag = batch['s1_mag'].to(device)
            s2_mag = batch['s2_mag'].to(device)
            is_supervised = batch['is_supervised'].to(device)
            
            # Forward pass
            pred_s1, pred_s2, masks = model(mix_mag)
            
            # Compute loss
            loss_dict = criterion(pred_s1, pred_s2, s1_mag, s2_mag, mix_mag, is_supervised)
            
            total_loss += loss_dict['total'].item()
            total_sup_loss += loss_dict['supervised']
            total_unsup_loss += loss_dict['unsupervised'].item()
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'supervised_loss': total_sup_loss / n_batches,
        'unsupervised_loss': total_unsup_loss / n_batches,
    }


def main(
    train_meta_path="dataset_stft/meta.jsonl",
    val_meta_path="dataset_stft/meta.jsonl",  # Use same for now, split later
    output_dir="checkpoints",
    supervised_ratio=1.0,  # 1.0 = fully supervised, 0.5 = 50% labeled
    batch_size=16,
    num_epochs=50,
    learning_rate=1e-3,
    lambda_unsup=0.1,
    num_workers=4,
    save_every=5,
):
    """Main training function"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_meta_path=train_meta_path,
        val_meta_path=val_meta_path,
        batch_size=batch_size,
        supervised_ratio=supervised_ratio,
        num_workers=0,  # Use 0 for CPU to avoid multiprocessing issues
    )
    
    # Create model
    print("\nCreating model...")
    model = ImprovedBSS(n_channels=2, n_freq=257, n_sources=2).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create loss and optimizer
    criterion = SemiSupervisedBSSLoss(lambda_unsup=lambda_unsup)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, accumulation_steps=4)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Log metrics
        print(f"\nEpoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_metrics['loss']:.4f} "
              f"(Sup: {train_metrics['supervised_loss']:.4f}, "
              f"Unsup: {train_metrics['unsupervised_loss']:.4f})")
        print(f"Val Loss: {val_metrics['loss']:.4f} "
              f"(Sup: {val_metrics['supervised_loss']:.4f}, "
              f"Unsup: {val_metrics['unsupervised_loss']:.4f})")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Loss/train_supervised', train_metrics['supervised_loss'], epoch)
        writer.add_scalar('Loss/train_unsupervised', train_metrics['unsupervised_loss'], epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Save checkpoint
        if epoch % save_every == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
    
    writer.close()
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train semi-supervised BSS model")
    parser.add_argument("--train_meta", default="dataset_stft/meta.jsonl")
    parser.add_argument("--val_meta", default="dataset_stft/meta.jsonl")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--supervised_ratio", type=float, default=1.0,
                        help="Fraction of labeled data (1.0=fully supervised)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_unsup", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5)
    
    args = parser.parse_args()
    
    main(
        train_meta_path=args.train_meta,
        val_meta_path=args.val_meta,
        output_dir=args.output_dir,
        supervised_ratio=args.supervised_ratio,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        lambda_unsup=args.lambda_unsup,
        num_workers=args.num_workers,
        save_every=args.save_every,
    )
