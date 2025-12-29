# AI4Alzheimer's Hack4Health Submission

## Overview
This project uses artificial intelligence to detect Alzheimer's using
* MRI brain scans
* Clinical and biomarker data 
* An interactive Streamlit dashboard
The goal of this product was to create a multimodal ML pipeline that would aid in research and early diagnostic exploration.

This repository was developed as part of Hack4Health: AI for ALzheimer's Research

## Models
### 1. MRI CNN Model
* Input: 128x128 grayscale MRI slices
* Model: Custom Convolutional Neural Network (TensorFlow)
* Output: 4-class classification 
    *(Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented)*

**Evaluation metrics**
* Accuracy
* Weighted F1
* Precision
* Sensitivity
* Confusion Matrix

### 2. Clinical/Biomarker Model
* Input: 
    * Amyloid PET biomarkers
    * Tau PET biomarkers
    * MMSE
    * CDR-SB
    * Demographics (gender, education)
* Model: XGBoost (multiclass)
* Output: 3-class classification
    *(CN-Cognitively Normal, MCI-Mildly Cognitive Impaired, AD-Alzheimer's Disease)*]

**Evaluation metrics**
* Weighted F1
* Precision
* Sensitivity
* Confusion Matrix

## Data Access and Ethics
### MRI Data
The MRI data in this project are from a publicly available dataset, given by Hack4Health.
Reference training scripts for preprocessing.

### Clinical Data
Clinical and biomarker data  are from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. Because of data-use restrictions, this data cannot be put in this repository.
In order to reproduce the clinical model's training:
1. Regrister for ADNI access
2. Download the required CSV files
3. Place them in
```models/clinical_model/data```

### How to Run
1. Install dependencies
```pip install -r requirements.txt```

2. Train models (optional)
This is optional, as the models are already saved in this repo. In order to train models, you must first gain access to all datasets described above.
```python training/train_mri.py```
```python training/train_clinical.py```

3. Evaluate models
```python evaluation/eval_mri.py```
```python evaluation/eval_clinical.py```
All metrics and plots from evaluation files are saved to
```results/```

4. Run dashboard
```streamlit run Dashboard.py```

### Reproducibility
* Random seeds are set for NumPy, TensorFlow, and PyTorch
* Train/val/test splits are fixed
* All evaluation metrics are computed on fixed test sets
* Final metrics are saved in ```results/```

### AI Tools Disclosure
Generative AI tools were only used for
* Code bugigng assistance
* Documentation suggesting
All modeling decision, data processing, analysis, and written submission were created by Leela Unnikrishnan.

### Limitations
* MRI images are 2D slices instead of the full 3D scan
* Class imbalances affect model's performance on minority classes
* Models are for research, and **not for professional medical use**

### Creator: Leela Unnikrishnan
