# Text-to-Image Model

This is a simple text-to-image generation model based on CLIP text encoder and a generative CNN.

## Model Architecture

The model consists of:
1. **Text Encoder**: Uses CLIP to encode text prompts into embeddings
2. **Image Generator**: A CNN-based generator that creates images from text embeddings
3. **Attention Mechanism**: Fuses text and image features using attention

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

This model uses the COCO Captions dataset. To download and prepare the dataset:

1. Download the COCO 2017 dataset:
   - Images: http://images.cocodataset.org/zips/train2017.zip
   - Annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

2. Extract the files to create the following directory structure:
```
coco/
├── train2017/
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
└── annotations/
    ├── captions_train2017.json
    └── ...
```

## Training

To train the model:

```bash
python train.py \
  --data_dir /path/to/coco/train2017 \
  --ann_file /path/to/coco/annotations/captions_train2017.json \
  --batch_size 32 \
  --epochs 50 \
  --learning_rate 0.0002 \
  --image_size 64 \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs
```

### Training Parameters

- `--data_dir`: Path to COCO images directory
- `--ann_file`: Path to COCO annotations file
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.0002)
- `--image_size`: Size of generated images (default: 64)
- `--checkpoint_dir`: Directory to save model checkpoints (default: ./checkpoints)
- `--log_dir`: Directory for TensorBoard logs (default: ./logs)

## Testing

To generate images from text prompts using a trained model:

```bash
python test.py \
  --model_path /path/to/checkpoint/model_final.pt \
  --output_dir ./generated_images
```

### Testing Parameters

- `--model_path`: Path to trained model checkpoint
- `--prompts_file`: Path to file with test prompts (one per line)
- `--output_dir`: Directory to save generated images (default: ./generated_images)

By default, the test script will generate images for these prompts:
- "a red apple on a wooden table"
- "a blue car driving on a road"
- "a cat sitting on a couch"
- "a dog playing in a park"
- "a mountain landscape with snow"
- "a beach with palm trees and ocean"

## Model Files

- `model.py`: Model architecture implementation
- `data.py`: Data loading and preprocessing
- `train.py`: Training script
- `test.py`: Testing/generation script
- `requirements.txt`: Required Python packages

## Notes

1. The model is a simplified implementation for demonstration purposes
2. For better results, consider using more sophisticated architectures like DALL-E or Stable Diffusion
3. Training on the full COCO dataset may require significant computational resources
4. You can limit the number of training samples using the `--max_samples` parameter during training