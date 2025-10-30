"""
Data processing for COCO Captions dataset
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO


class CocoCaptionsDataset(Dataset):
    """COCO Captions dataset"""
    
    def __init__(self, root_dir, ann_file, transform=None, max_samples=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            ann_file (str): Path to the json annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_samples (int, optional): Maximum number of samples to load.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load annotations
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())
        
        # Limit samples if specified
        if max_samples is not None:
            self.ids = self.ids[:max_samples]
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        
        # Get image id and caption
        img_id = ann['image_id']
        caption = ann['caption']
        
        # Load image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root_dir, path)).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, caption


def get_transforms(img_size=64):
    """Get image transformations"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    return transform


def get_dataloader(data_dir, ann_file, batch_size=32, img_size=64, max_samples=None, num_workers=4):
    """Get COCO captions dataloader"""
    transform = get_transforms(img_size)
    
    dataset = CocoCaptionsDataset(
        root_dir=data_dir,
        ann_file=ann_file,
        transform=transform,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def download_coco_dataset():
    """Instructions for downloading COCO dataset"""
    instructions = """
    To download the COCO Captions dataset, follow these steps:
    
    1. Download the dataset from http://cocodataset.org/#download
       - For images: 2017 Train images [118K/18GB]
       - For annotations: 2017 Train annotations [241MB]
       
    2. Extract the files to a directory structure like:
       coco/
       ├── train2017/
       │   ├── 000000000009.jpg
       │   ├── 000000000025.jpg
       │   └── ...
       └── annotations/
           ├── captions_train2017.json
           └── ...
           
    3. Update the paths in the training script accordingly.
    """
    print(instructions)


if __name__ == "__main__":
    download_coco_dataset()