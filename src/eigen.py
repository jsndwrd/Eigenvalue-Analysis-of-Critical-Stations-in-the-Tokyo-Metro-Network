import numpy as np
from .qr import householder

def qrEigen(A: np.ndarray, max_iter=1000, tol=1e-10):
    Ak = A.astype(float).copy()
    n = Ak.shape[0]
    Qt = np.eye(n)

    for _ in range(max_iter):
        Q, R = householder(Ak)
        An = R @ Q
        Qt = Qt @ Q

        offD = An - np.diag(np.diag(An))
        if np.linalg.norm(offD, ord="fro") < tol:
            Ak = An
            break

        Ak = An

    value = np.diag(Ak)
    vector = Qt
    return value, vector

def spectralRadius(A: np.ndarray):
    value = np.linalg.eigvals(A) # Use library since its fasterrrrr
    value = np.asarray(value)
    return np.max(np.abs(value))
