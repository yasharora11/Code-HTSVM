# Code-HTSVM
This repository contains the implementation of the HTSVM model entitled "A robust twin support vector machine with huberized hinge loss function with an application in handwritten digit recognition". It is designed to handle binary classification problems in real-world settings. The script `Main.py` demonstrates the method using the Hepatitis dataset and evaluates performance via 10-fold cross-validation. To apply the model to other datasets used in the paper, simply update the dataset path in `Main.py`. The repository is organized into separate folders for:
1. analysis: It includes scripts for evaluating the model under different hyperparameter settings;
2. plots: It provides code for generating the 3D surface and bar plots included in the paper;
3. data: It contains all the datasets used in the experiments.

## Files
- `HTSVM_algo.py`: Core model implementation (training and prediction)
- `Main.py`: Script to run 10-fold CV, compute metrics, and print summary

## Usage
1. Install the required dependencies.
2. Import or place your dataset in the data/ folder.
3. Run the script: python Main.py
