import numpy as np

def householder(A: np.ndarray):
    A = A.astype(float).copy()
    m, n = A.shape
    Q = np.eye(m)

    for k in range(n):
        x = A[k:, k]
        if np.allclose(x[1:], 0):
            continue

        e1 = np.zeros_like(x)
        e1[0] = 1.0
        alpha = np.linalg.norm(x)
        if x[0] >= 0:
            alpha = -alpha
        u = x - alpha * e1
        v = u / np.linalg.norm(u)

        Hk = np.eye(m)
        Hk[k:, k:] -= 2.0 * np.outer(v, v)

        A = Hk @ A
        Q = Q @ Hk

    R = A
    return Q, R
