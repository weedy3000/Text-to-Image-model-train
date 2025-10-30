"""
Training script for text-to-image model
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from text2image.model import TextToImageModel
# from text2image.data import get_dataloader
from improve_t2i_model import TextToImageModel

def save_model(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Model saved to {filepath}")


def train(args):
    """Train the text-to-image model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = TextToImageModel(
        img_size=args.image_size,
        channels=args.channels
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.MSELoss()
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Get dataloader
    print("Loading dataset...")
    dataloader = get_dataloader(
        data_dir=args.data_dir,
        ann_file=args.ann_file,
        batch_size=args.batch_size,
        img_size=args.image_size,
        max_samples=args.max_samples
    )
    
    print(f"Dataset loaded with {len(dataloader.dataset)} samples")
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, captions) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            generated_images = model(captions)
            
            # Calculate loss
            loss = criterion(generated_images, images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Log progress
            if batch_idx % args.log_interval == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
                writer.add_scalar('Batch/Loss', loss.item(), global_step)
            
        # Average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch}/{args.epochs}] Average Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('Epoch/Average_Loss', avg_epoch_loss, epoch)
        
        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            save_model(model, optimizer, epoch, avg_epoch_loss, checkpoint_path)
            
            # Save sample images
            model.eval()
            with torch.no_grad():
                sample_captions = ["a cat sitting on a chair", "a dog playing in the park"]
                sample_images = model(sample_captions)
                
                # Save images to tensorboard
                img_grid = torch.cat([img.unsqueeze(0) for img in sample_images[:4]], dim=0)
                writer.add_images('Generated_Samples', img_grid, epoch)
                
            model.train()
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "model_final.pt")
    save_model(model, optimizer, args.epochs-1, avg_epoch_loss, final_path)
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Text-to-Image Model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COCO images directory")
    parser.add_argument("--ann_file", type=str, required=True, help="Path to COCO annotations file")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to use")
    
    # Model parameters
    parser.add_argument("--image_size", type=int, default=256, help="Size of generated images")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    
    # Logging and saving
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train(args)