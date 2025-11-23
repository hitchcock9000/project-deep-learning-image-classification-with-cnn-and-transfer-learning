# Deep Learning Image Classification Project

## Project Overview
This project implements image classification on the CIFAR-10 dataset using two approaches:
1. **Custom CNN**: A convolutional neural network built from scratch
2. **Transfer Learning**: Using pre-trained models (VGG16, ResNet50, MobileNetV2)

## Project Structure
```
.
├── notebooks/
│   ├── 1_data_exploration.ipynb    # Data loading and visualization
│   ├── 2_custom_cnn.ipynb          # Custom CNN implementation
│   ├── 3_transfer_learning.ipynb   # Transfer learning implementation
│   └── 4_comparison.ipynb          # Model comparison
├── data/                           # Saved models and results
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Dataset
**CIFAR-10** consists of 60,000 32x32 color images in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images
- 10,000 test images

Dataset location: `/Users/nim/Downloads/cifar-10-batches-py`

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Dataset
Ensure the CIFAR-10 dataset is available at the specified path.

## Usage

### Using Jupyter Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` directory

3. Run notebooks in order:
   - `1_data_exploration.ipynb` - Explore the dataset
   - `2_custom_cnn.ipynb` - Train custom CNN
   - `3_transfer_learning.ipynb` - Train transfer learning model
   - `4_comparison.ipynb` - Compare results

## Model Architectures

### Custom CNN
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch Normalization after each conv layer
- MaxPooling and Dropout for regularization
- Fully connected layers with 256 units
- Total parameters: ~1.5M

### Transfer Learning
- Pre-trained base (frozen weights)
- Global Average Pooling
- Dense layer (256 units)
- Dropout for regularization
- Fine-tuned for CIFAR-10

## Training Details

### Data Preprocessing
- Normalization: Pixel values scaled to [0, 1]
- Train/Val split: 80/20
- Data augmentation:
  - Random rotation (±15°)
  - Width/height shift (±10%)
  - Horizontal flip

### Training Configuration
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch size: 64
- Early stopping: Patience of 10 epochs
- Model checkpointing: Save best model based on validation accuracy

## Expected Results

### Custom CNN
- Expected accuracy: 70-75%
- Training time: ~30-40 minutes (CPU), ~5-10 minutes (GPU)

### Transfer Learning (MobileNetV2)
- Expected accuracy: 75-85%
- Training time: ~20-30 minutes (CPU), ~3-5 minutes (GPU)

## Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision
- **Recall**: Per-class recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run data exploration**: Open `notebooks/1_data_exploration.ipynb`
3. **Train models**: Run the custom CNN and transfer learning scripts
4. **Compare results**: Use `compare_models.py` or the comparison notebook
5. **Prepare presentation**: Document findings and create PPT

## Tips for Better Results

1. **Increase epochs**: If training stops early, increase the patience parameter
2. **Try different architectures**: Experiment with more/fewer layers
3. **Adjust learning rate**: Try different learning rates (0.0001, 0.001, 0.01)
4. **More augmentation**: Add more aggressive data augmentation
5. **Fine-tuning**: Unfreeze some layers of the pre-trained model for fine-tuning

## Troubleshooting

### Out of Memory
- Reduce batch size to 32 or 16
- Use a lighter model (MobileNetV2 instead of VGG16)

### Slow Training
- Use GPU if available
- Reduce number of epochs
- Use a smaller model

### Poor Accuracy
- Check data preprocessing
- Increase model complexity
- Add more data augmentation
- Train for more epochs

## Team Members
- Natasha Silvestre

## License
This project is for educational purposes as part of the Ironhack bootcamp.