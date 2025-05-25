import numpy as np
import pandas as pd
import HTSVM_algo
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
import statistics as st

# Configuration
DATA_PATH = 'data/Hepatitis.xlsx'
K_FOLDS = 10
C=1  # selected from [10^-2,...,10^2]
lambda_val = 0.1  # selected from [0.1,0.2,...,9]
delta=0.001   # selected from [10^-4,...,10^-1]

def load_data(path):
    df = pd.read_excel(path, header=None)
    return df.to_numpy()

def evaluate_htsvm(X, C,lambda_val,delta):
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=30)
    metrics = {'acc': [], 'sens': [], 'spec': [], 'precision': [], 'f1': [], 'mcc': []}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = X_train[:, -1], X_test[:, -1]

        A_train = X_train[y_train == 1][:, :-1]
        B_train = X_train[y_train == -1][:, :-1]
        A_test = X_test[y_test == 1][:, :-1]
        B_test = X_test[y_test == -1][:, :-1]

    
        # Normalize both classes independently
        scaler = MinMaxScaler()
        A_train = scaler.fit_transform(A_train)
        A_test = scaler.transform(A_test)

        B_train = scaler.fit_transform(B_train)
        B_test = scaler.transform(B_test)

        w1, b1 = HTSVM_algo.fit(A_train, B_train, -1,C,lambda_val,delta)
        w2, b2 = HTSVM_algo.fit(B_train, A_train, 1,C,lambda_val,delta)
        w = np.c_[w1.T, w2.T]
        b = np.c_[b1, b2]

        pred1 = HTSVM_algo.predict(A_test, w, b)
        pred2 = HTSVM_algo.predict(B_test, w, b)
        preds = np.r_[pred1, pred2]
        y_true = np.r_[np.ones(len(pred1)), -np.ones(len(pred2))]

        metrics['acc'].append(accuracy_score(y_true, preds) * 100)
        metrics['f1'].append(f1_score(y_true, preds) * 100)
        metrics['mcc'].append(matthews_corrcoef(y_true, preds))

        cm = confusion_matrix(y_true, preds)
        TP, FN, FP, TN = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
        metrics['sens'].append(TP / (TP + FN) * 100)
        metrics['spec'].append(TN / (TN + FP) * 100)
        metrics['precision'].append(TP / (TP + FP) * 100)

    return metrics

def main():
    X = load_data(DATA_PATH)
    metrics = evaluate_htsvm(X, C,lambda_val,delta)

    results_df = pd.DataFrame({
        'avg_acc_score': [np.mean(metrics['acc'])],
        'standard_deviation': [st.stdev(metrics['acc'])],
        'MCC': [np.mean(metrics['mcc'])],
        'sensitivity': [np.mean(metrics['sens'])],
        'specificity': [np.mean(metrics['spec'])],
        'precision': [np.mean(metrics['precision'])],
        'F_measure': [np.mean(metrics['f1'])]
    })

    print("\nFinal Results Summary:")
    print(results_df.round(2))

if __name__ == "__main__":
    main()
