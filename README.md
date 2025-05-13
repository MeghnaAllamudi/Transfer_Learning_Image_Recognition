# Transfer Learning Image Recognition

## Project Overview
This repository contains our work on transfer learning for image recognition. The project involves multiple approaches to solve a multi-class image recognition problem.

## Repository Contents

- `ma4407_approaches.ipynb`: This notebook contains early approaches and experiments by team member ma4407. It includes various model architectures and techniques that were tried during development.

- `sb4815_trial-n-error.ipynb`: This notebook contains different approaches and experiments by team member sb4815. It includes various model configurations and techniques that were explored.

- `COMS_4995_Final_Best_Solution.ipynb`: This is our final solution notebook with the highest-performing model. This approach achieved the best results and should be used as the reference implementation.

- `final_predictions.csv`: This file contains our final predictions for the test dataset.

## Running the Final Solution

To run our final solution in your own environment, follow these steps:

### Adapting File Paths

The notebook `COMS_4995_Final_Best_Solution.ipynb` needs to be modified in the first few cells to point to your local data directories:

1. **Update Data Paths**: Locate the following code block in the notebook (usually within the first 30 cells):

```python
train_ann_df = pd.read_csv('/content/drive/My Drive/train_data.csv')
super_map_df = pd.read_csv('/content/drive/My Drive/superclass_mapping.csv')
sub_map_df = pd.read_csv('/content/drive/My Drive/subclass_mapping.csv')

train_img_dir = '/content/drive/My Drive/train_images/train_images/'
test_img_dir = '/content/drive/My Drive/test_images/test_images/'
```

Replace these paths with the location of your data files:

```python
train_ann_df = pd.read_csv('YOUR_PATH/train_data.csv')
super_map_df = pd.read_csv('YOUR_PATH/superclass_mapping.csv')
sub_map_df = pd.read_csv('YOUR_PATH/subclass_mapping.csv')

train_img_dir = 'YOUR_PATH/train_images/'
test_img_dir = 'YOUR_PATH/test_images/'
```

2. **Cache Directory**: If your environment has limited memory, you might want to adjust the local cache directory:

```python
local_cache_dir = "/content/local_train_image_cache"
```

Change this to a location that works for your setup.

### Required Dependencies

Make sure you have the following packages installed:

- PyTorch
- Pandas
- NumPy
- Pillow
- tqdm
- Matplotlib

### Running the Model

After adjusting the file paths, you can run the notebook cells in sequence. The final solution implements:

1. Data loading and preprocessing
2. A CNN-based model architecture
3. Training with knowledge distillation
4. Novelty detection for unknown classes
5. Final prediction generation

## Final Predictions

The `final_predictions.csv` file in this repository contains our predictions for the test dataset. These are the final results submitted for evaluation.

## Approach Description

Our final solution uses a CNN-based architecture with:

- Transfer learning from a pre-trained model
- Fine-tuning on the target dataset
- Energy-based novelty detection for identifying unknown classes
- Hierarchical classification (superclass and subclass levels)

The novelty detection component was crucial for handling unknown classes that weren't seen during training.

## Contact

For any questions about this repository or implementation details, please open an issue in the repository.