# Unitary Matrix Product States Using Riemannian Optimization (Code)

This repository provides the **MATLAB implementation** for the paper:

**Efficient Generative Modeling with Unitary Matrix Product States Using Riemannian Optimization**

The code implements a generative modeling framework based on **unitary Matrix Product States (MPS)** and **Riemannian optimization**, and is used for experiments on datasets including Bars and Stripes, and EMNIST (includes the MNIST dataset).

---

## Requirements

- MATLAB (R2022a or later is recommended)
- No additional toolboxes are required

---

## Running the Code

To run the code, please follow the steps below:

1. Add required folders to the MATLAB path

   - datasets
   - figure
   
   Alternatively, run the following commands in MATLAB:
   ```matlab
   addpath(genpath('datasets'));
   addpath(genpath('figure'));
   ```
   
2. Run the main script

   Execute:
   ```matlab
   main_generate_sample_mnist.m
   ```

   This script trains the unitary MPS generative model and generates samples for the MNIST dataset.

---

## Dataset Notes

### 1. Bars and Stripes Dataset

- The Bars and Stripes (BAS) dataset is located in the `datasets` folder.
- The dataset can be generated using:
  ```matlab
  % H -> height, W -> width
  generate_bas(H,W) 
  ```
- Dataset parameters can be adjusted inside the function as needed.

---

### 2. EMNIST Dataset

- The EMNIST dataset can be downloaded from:

  https://www.nist.gov/itl/products-and-services/emnist-dataset

- After downloading, use the following function to read and preprocess the dataset:
  ```matlab
  save_images_emnist.m
  ```

---

## Comparison Experiments

The comparison experiments reported in the paper can be found in the following repository:

https://github.com/congzlwag/UnsupGenModbyMPS

This repository provides baseline implementations for unsupervised generative modeling using Matrix Product States.

---

## Citation

If you use this code in your research, please cite the corresponding paper.

---

Authors:Haotong Duan, Zhongming Chen, Ngai Wong.
