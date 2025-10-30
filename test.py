"""
Test script for text-to-image model
"""

import os
import argparse
import torch
from torchvision.utils import save_image
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from text2image.model import TextToImageModel
from improve_t2i_model import TextToImageModel

def load_model(model_path, device,image_size,channels):
    """Load trained model"""
    model = TextToImageModel(
        img_size=image_size,
        channels=channels
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def generate_images(model, texts, device, output_dir):
    """Generate images from text prompts"""
    with torch.no_grad():
        # Generate images
        generated_images = model(texts)
        
        # Save images
        for i, (text, img) in enumerate(zip(texts, generated_images)):
            # Denormalize image from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            # Save image
            filename = f"generated_{i}.png"
            filepath = os.path.join(output_dir, filename)
            save_image(img, filepath)
            print(f"Saved generated image for '{text}' to {filepath}")


def test_model(args):
    """Test the text-to-image model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test prompts
    if args.prompts_file:
        # Load prompts from file
        with open(args.prompts_file, 'r') as f:
            test_prompts = [line.strip() for line in f.readlines()]
    else:
        # Default test prompts
        test_prompts = [
            "a red apple on a wooden table",
            "a blue car driving on a road",
            "a cat sitting on a couch",
            "a dog playing in a park",
            "a mountain landscape with snow",
            "a beach with palm trees and ocean"
        ]
    
    print("Generating images for prompts:")
    for prompt in test_prompts:
        print(f"  - {prompt}")
    
    # Generate images
    generate_images(model, test_prompts, device, args.output_dir)
    
    print(f"All images saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Text-to-Image Model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to file with test prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="./generated_images", help="Output directory for generated images")
    parser.add_argument("--image_size", type=int, default=256, help="Size of generated images")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    args = parser.parse_args()
    
    test_model(args)