# CADHE: Privacy-Preserving Medical Image Analysis Through Homomorphic Encrypted Convolutional Networks
Stefano Cirillo, Vincenzo Deufemia, Luigi Di Biasi, Giuseppe Polese, Giandomenico Solimando, Genoveffa Tortora  

---

## Abstract

Hospitals and clinical centres must comply with strict privacy and security regulations, which makes it hard to share medical data and collaboratively train AI models. CADHE is a lightweight convolutional neural network specifically designed to work directly on homomorphically encrypted MRI images of multiple sclerosis patients. The network is instantiated using the CKKS approximate homomorphic encryption scheme and supports end-to-end encrypted inference without exposing patient data at any stage of the pipeline.

We evaluate CADHE on the task of predicting the Expanded Disability Status Scale (EDSS) for multiple sclerosis patients from multi-prospective MRI scans (T1, T2 and FLAIR). CADHE is tested on both binary (EDSS ≤ 2 vs. EDSS > 2) and multi-class (normal, mild, severe) classification. Despite operating entirely in the encrypted domain, CADHE reaches accuracy values comparable to the plaintext counterpart, achieving around 88% accuracy on the all-modalities dataset and 83% on T1 for the binary task, and up to 82% in multi-class classification. These results show that carefully designed encrypted convolutional networks can provide strong privacy guarantees while maintaining clinically useful performance.

---

## Project Overview

This repository contains the code underlying CADHE, a privacy-preserving CNN for EDSS classification from MRI data using homomorphic encryption (HE).

### Main Features

- **End-to-end encrypted inference**  
  Convolutional and fully connected layers are evaluated on CKKS-encrypted tensors (via TenSEAL). Raw MRI slices are never exposed in plaintext on the server side.

- **Multiple sclerosis severity prediction**  
  - **Binary task**:  
    - EDSS ≤ 2.0 → no/minimal disability  
    - EDSS > 2.0 → presence of neurological impairment  
  - **Multi-class task**:  
    - Normal: EDSS ≤ 2.0  
    - Mild: 2.0 < EDSS ≤ 4.0  
    - Severe: EDSS > 4.0  

- **Plaintext vs encrypted models**  
  The plaintext CNN and its fully encrypted counterpart share the same architecture and weights, enabling a direct comparison between standard and HE-based inference.
 ---
### Dataset

The experiments rely on a publicly available Brain MRI dataset of multiple sclerosis with consensus manual lesion segmentation and patient meta-information.

The dataset is **not** redistributed in this repository.  
To obtain the data, please refer to:

```
@article{muslim2022brain,
  title        = {Brain MRI dataset of multiple sclerosis with consensus manual lesion segmentation and patient meta information},
  author       = {Muslim, Ali M and Mashohor, Syamsiah and Al Gawwam, Gheyath and Mahmud, Rozi and binti Hanafi, Marsyita and Alnuaimi, Osama and Josephine, Raad and Almutairi, Abdullah Dhaifallah},
  journal      = {Data in Brief},
  volume       = {42},
  pages        = {108139},
  year         = {2022},
  publisher    = {Elsevier}
}
```

By using this dataset together with the code in this repository, you agree to follow the dataset’s original license and citation requirements.

---
### CADHE Architecture

The CADHE network is a compact CNN tailored to the constraints of homomorphic encryption (limited depth, polynomial activations, small input size). MRI slices are pre-processed and downsampled to **1 × 32 × 32** greyscale images before encryption.

![CADHE Architecture](images/cadhe_architecture.png)

The architecture (see the corresponding figure in the paper) can be summarised as follows:

- **Input**  
  - 1 × 32 × 32 greyscale MRI slice (encrypted).

- **Encrypted Convolutional Block**  
  - HE-Conv2D: 1 → 4 channels, kernel size 7 × 7, stride 3, padding 0  
  - Quadratic activation applied element-wise: \( f(x) = x^2 \)

- **Flattening Layer**  
  - Feature maps 4 × 9 × 9 flattened to a 324-dimensional vector.

- **Fully Connected Layers (encrypted)**  
  1. FC1: 324 → 32, followed by quadratic activation  
  2. FC2: 32 → 16, followed by quadratic activation  

- **Output Heads**  
  - **Binary head**  
    - FC: 16 → 2 logits  
    - Class mapping:  
      - logit 0 → EDSS ≤ 2  
      - logit 1 → EDSS > 2  
  - **Multi-class head**  
    - FC: 16 → 3 logits  
    - Class mapping:  
      - Class 0 → Normal (EDSS ≤ 2)  
      - Class 1 → Mild (2 < EDSS ≤ 4)  
      - Class 2 → Severe (EDSS > 4)  

All nonlinearities are implemented as quadratic functions, which are compatible with CKKS and avoid non-HE-friendly activations such as ReLU or sigmoid.

---
###  Application scenario with real data
![Scenario](images/Scenario.png)

---
### Experimental configuration

The experimental setup is defined as follows:

- **Input slice resolution**  
  Single–channel MRI slices are preprocessed and downsampled to **1 × 32 × 32** pixels.

- **Network depth and architecture**  
  - 1 convolutional layer with **4 filters**, kernel size **7 × 7**, stride **3**, no padding.  
  - 2 fully connected hidden layers with **32** and **16** neurons, respectively.  
  - A **quadratic activation** `f(x) = x^2` is applied after the convolutional layer and after each hidden fully connected layer (both in plaintext and encrypted models).

- **Hyperparameter grid**  
  - **Learning rate**: `{1e-2, 1e-3, 1e-4}`  
  - **Batch size**: `{32, 64}`  
  - **Hidden layer sizes**: fixed to **32** (first FC layer) and **16** (second FC layer).  
  - **Input image size**: fixed to **32 × 32** pixels.

- **Model selection**  
  A grid search evaluates all combinations of learning rate and batch size; the best configuration is selected based on validation performance.

- **Reproducibility**  
  Random seeds for Python / NumPy / PyTorch are exposed in the training scripts so that experiments can be made reproducible (see the corresponding configuration or script options for how to set them).

- **Software and hardware**  
  - Python **3.9**, PyTorch **2.7.1**, TensorFlow **2.19.0**, Scikit-learn **1.6.1**, CUDA **12.6**.  
  - Experiments were executed on a workstation with an **Intel i9 CPU @ 5 GHz (14 cores)**, **64 GB RAM**, and an **NVIDIA 3060 GPU**.
---
### HE parameters

The homomorphic encrypted version of CADHE uses a CKKS context created via TenSEAL with the following settings:

- **Scheme**: CKKS (approximate arithmetic).  
- **Polynomial modulus degree**: **16 384**.  
- **Coefficient modulus bit sizes**:  
  `[60, 26, 26, 26, 26, 26, 26, 26, 26, 26, 60]`
- **Global scale**:  
  A global scaling factor of `2^26` is used for fixed-point encoding of real numbers.

- **Keys**:  
  **Galois keys** are generated to enable the ciphertext rotations required by the packed implementation of convolutions and fully connected layers.

- **Packing strategy**:  
  Each **32 × 32** input slice is **flattened and encoded into a CKKS ciphertext**.  
  Convolution is implemented homomorphically by applying slot rotations and weighted additions over the packed slots, exploiting the SIMD nature of CKKS to process all patch positions in parallel.

---


## Requirements and Setup

The original experiments were run with:

- Python 3.9  
- PyTorch 2.7.1  
- CUDA 12.6  
- Scikit-learn 1.6.1  
- TensorFlow 2.19.0  
- TenSEAL (CKKS)

All core dependencies are listed in `requirements.txt`.

### 1. Clone the repository

```bash
git clone <URL-of-this-repository>.git
cd <repository-folder>
```

### 2. Creating a Python Virtual Environment

From within the repository folder:

```bash
# Create a virtual environment (Python 3.9 recommended)
python3 -m venv .venv

# Activate the environment (Linux/macOS)
source .venv/bin/activate

# On Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

With the virtual environment activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**  
> Depending on your hardware/OS, you may need to adapt library versions (especially `torch` and `tensorflow`) and install a CUDA toolkit compatible with your GPU.

### 4. Dataset preparation (high level)

1. Request / download the dataset from Muslim et al. (see the citation above).  
2. Apply the preprocessing described in the paper: skull-stripping, resizing, slice extraction, greyscale conversion, normalisation, and downsampling to 32 × 32 for CADHE.  
3. Organise the dataset on disk according to the structure expected by the training and evaluation scripts (for instance, per-modality folders and train/val/test splits).

---

## How to cite

By using this dataset and the accompanying code, you agree to cite the following article (fill in the final bibliographic details once available):

```bibtex
@inproceedings{cirillo2025cadhe,
  title={CADHE: privacy-preserving medical image analysis through homomorphic encrypted convolutional networks},
  author={Cirillo, Stefano and Deufemia, Vincenzo and Di Biasi, Luigi and Polese, Giuseppe and Solimando, Giandomenico and Tortora, Genoveffa},
  booktitle={2025 IEEE International Conference on Big Data (BigData)},
  pages={7064--7072},
  year={2025},
  organization={IEEE}
}
```
