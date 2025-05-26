import numpy as np
import pandas as pd
import HTSVM_algo
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import os

# Config
DATA_PATH = 'data/Hepatitis.xlsx'
OUTPUT_FILE = 'data/HTSVM_varying_delta_lambda.xlsx'
C = 1
lambda_list = np.arange(0.1, 9, 0.1)  # [0.1, 0.2, ..., 9]
delta_list = [10**(-i) for i in range(1, 5)]  # [0.1, 0.01, 0.001, 0.0001]
K_FOLDS = 10

def load_data(path):
    df = pd.read_excel(path, header=None)
    return df.to_numpy()

def get_accuracy(X, C, lambda_val, delta):
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=30)
    accs = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = X_train[:, -1], X_test[:, -1]

        A_train = X_train[y_train == 1][:, :-1]
        B_train = X_train[y_train == -1][:, :-1]
        A_test = X_test[y_test == 1][:, :-1]
        B_test = X_test[y_test == -1][:, :-1]

        scaler = MinMaxScaler()
        A_train = scaler.fit_transform(A_train)
        A_test = scaler.transform(A_test)
        B_train = scaler.fit_transform(B_train)
        B_test = scaler.transform(B_test)

        w1, b1 = HTSVM_algo.fit(A_train, B_train, -1, C, lambda_val, delta)
        w2, b2 = HTSVM_algo.fit(B_train, A_train, 1, C, lambda_val, delta)

        w = np.c_[w1.T, w2.T]
        b = np.c_[b1, b2]

        pred1 = HTSVM_algo.predict(A_test, w, b)
        pred2 = HTSVM_algo.predict(B_test, w, b)
        preds = np.r_[pred1, pred2]
        y_true = np.r_[np.ones(len(pred1)), -np.ones(len(pred2))]

        accs.append(accuracy_score(y_true, preds) * 100)

    return np.mean(accs)

def main():
    X = load_data(DATA_PATH)
    records = []

    for delta in delta_list:
        for lambda_val in lambda_list:
            acc = get_accuracy(X, C, lambda_val, delta)
            records.append({'delta': delta, 'lambda': lambda_val, 'accuracy': acc})
            print(f"delta = {delta:.4f}, lambda = {lambda_val:.1f}, accuracy = {acc:.2f}")

    df = pd.DataFrame(records)
    os.makedirs('results', exist_ok=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nSaved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
