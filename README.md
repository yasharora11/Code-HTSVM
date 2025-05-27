# Code-HTSVM

This repository contains the implementation of the model discussed in the article entitled "A robust twin support vector machine with huberized hinge loss function with an application in handwritten digit recognition". The script `Main.py` demonstrates the code using the Hepatitis dataset and evaluates its performance via 10-fold cross-validation. To apply the model to other datasets used in the article, update the dataset path (using the dataset from the folder "data") in `Main.py`. `HTSVM_algo.py` is used while running `Main.py`.

The repository is organized into separate folders for:
1. analysis: It includes scripts for evaluating the model under different hyperparameter settings;
2. plots: It provides code for generating the 3D surface and bar plots included in the article;
3. data: It contains all the datasets used in the experiments.

## Files
- `HTSVM_algo.py`: Core model implementation (training and prediction)
- `Main.py`: Script to run 10-fold CV, compute metrics, and print summary

## Usage
1. Install the required dependencies.
2. Import or place your dataset in the data/ folder.
3. Run the script: python Main.py
