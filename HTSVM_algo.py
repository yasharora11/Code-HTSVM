import numpy as np
from numpy import linalg as LA

def fit(A, B, y, C,lambda1,delta):
    """
    Fits a model to training data
    
    Parameters:
    A : Patterns from the positive class (+1).
    B : Patterns from the negative class (-1).
    y : Label (+1 or -1).
    C : Penalty parameter
    lambda1 : Regularization parameter for L1 norm.
    lambda2, lambda3: Initialize to 1 

    Returns:
    w : Weight vector of the classifier.
    b : Bias term of the classifier.
    """
    
    m, m2 = len(B), len(A)
    n = A.shape[1]
    e = np.ones(m2)

    # Regularization constants and initialization
    lambda2, lambda3 = 1, 1
    t0 = 1
    p = []
    b0 = 0
    w0 = np.zeros(n)
    bcap, wcap = b0, w0
    F0 = m * (1 - delta / 2)

    # Lipschitz constants
    L_f1 = (C / delta) * sum([1 + np.inner(B[i], B[i]) for i in range(m)])
    L_f2 = sum([1 + np.inner(A[i], A[i]) for i in range(m2)])
    L_f=L_f1+L_f2
    L0 = (2 *L_f) / n

    cn, k = 0, 0

    while cn < 3:
        k += 1
        t = 0.5 * np.sqrt(1 + 4 * t0**2)

        # Gradient computations
        f1gradb = np.zeros(m)
        f1gradw = np.zeros((m, n))
        f2gradb = np.zeros(m2)
        f2gradw = np.zeros((m2, n))

        for i in range(m):
            margin = y * (bcap + B[i] @ wcap)
            if margin >= 1:
                continue
            elif (1 - delta) <= margin < 1:
                f1gradb[i] = -y * (1 - margin) / delta
                f1gradw[i] = -y * B[i] * (1 - margin) / delta
            else:
                f1gradb[i] = -y
                f1gradw[i] = -y * B[i]

        for i in range(m2):
            margin = bcap + A[i] @ wcap
            f2gradb[i] = 2 * margin
            f2gradw[i] = 2 * A[i] * margin

        L = 1.2 * L0

        # Bias update
        b = (L * bcap - f1gradb.sum() - f2gradb.sum()) / (L + lambda3)

        # Weight update via soft-thresholding (proximal operator)
        s1 = L * wcap - f1gradw.sum(axis=0) - f2gradw.sum(axis=0)
        s2 = np.maximum(np.abs(s1) - lambda1, 0)
        s = np.sign(s1) * s2
        w = s / (L + lambda2)

        # Objective function
        phi = np.array([
            0 if (m1 := y * (b + B[j] @ w)) >= 1 else
            ((1 - m1)**2) / (2 * delta) if (1 - delta) <= m1 < 1 else
            1 - m1 - delta / 2
            for j in range(m)
        ])
        f1 = phi.sum()
        f2 = (1 / 2) * sum([(b + A[j] @ w)**2 for j in range(m2)])
        f = (C *f1) + f2

        g = lambda1 * LA.norm(w, 1) + (lambda2 / 2) * LA.norm(w)**2 + (lambda3 / 2) * b**2
        F = f + g

        omega = min(t0 / t, np.sqrt(L0 / L))
        if F > F0:
            bcap, wcap = b, w
        else:
            bcap = b + omega * (b - b0)
            wcap = w + omega * (w - w0)

        u0 = np.r_[b0, w0]
        u = np.r_[b, w]
        if (F0 - F) / (1 + F0) <= 1e-6 and LA.norm(u0 - u) / (1 + LA.norm(u0)) <= 1e-6:
            p.append(k)
            if len(p) >= 3 and p[-1] - p[-2] == p[-2] - p[-3] == 1:
                cn = 3

        t0, F0, L0, b0, w0 = t, F, L, b, w

    return w, b


def predict(X, w, b):
    """
    Predicts labels for input data X using the trained model.
    
    Parameters:
    X : Input samples.
    w : Weight matrix (2 columns, one per class).
    b : Bias terms for each class.

    Returns:
    pred : Predicted class labels (+1 or -1).
    """
    pred = np.zeros(len(X))
    for i in range(len(X)):
        d = [abs(X[i] @ w[:, r] + b[:, r]) / LA.norm(w[:, r]) for r in range(2)]
        pred[i] = -1 if np.argmin(d) == 1 else 1
    return pred
