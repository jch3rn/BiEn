# BiEn: Bilevel Ensemble Learning for Semi-Supervised Medical Image Segmentation

## Installation

```bash
conda env create -f environment.yml
conda activate BiEn
```

## Dataset Preparation

1. **Download the datasets**:  
   - [Fetoscopic Dataset](https://drive.google.com/file/d/1gVBtBsFHj--uCtqBHpgoVHETYpjfMkHm/view?usp=sharing)  
   - [Endoscopic Dataset](https://drive.google.com/file/d/14S6-5Q3abFg61a6riKO8TQ8T1TN6kuhi/view?usp=sharing)  
   - [Lung Nodule Dataset](https://drive.google.com/file/d/1Sdxv3XwbhSpFSVv8mOQaW_ds6du3tDlu/view?usp=sharing)  

2. **Extract all archives** directly into the `data/` directory. Ensure the extracted folders match the following structure:  
   ```bash
   BiEn/
   ├── data/
   │   ├── endo/
   │   ├── feto/
   │   └── lung/
   └── code/
       └── ...
    ```

## Training

1. **Train on the Fetoscopic Dataset**:
   ```
   python train.py --root_path ../data/feto --exp feto/BiEn --num_classes 4
   ```

2. **Train on the Endoscopic Dataset**:
   ```
   python train.py --root_path ../data/endo --exp endo/BiEn --num_classes 7
   ```

3. **Train on the Lung Nodule Dataset**:
   ```
   python train.py --root_path ../data/lung --exp lung/BiEn --num_classes 2
   ```

## Testing
   ```
   python test.py
   ```

## Acknowledgements
Our implementation is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/) and [PS-MT](https://github.com/yyliu01/PS-MT). We sincerely thank the authors for their contributions.