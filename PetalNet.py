import os
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ─────── Data Loading & Preprocessing ───────
iris = load_iris()
X = iris.data               # (150, 4)
y = iris.target.reshape(-1, 1)

# one-hot encode labels → (150, 3)
encoder    = OneHotEncoder(sparse_output=False)
y_encoded  = encoder.fit_transform(y)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# feature‐scale inputs to zero‑mean/unit‑variance
scaler      = StandardScaler()
X_train     = scaler.fit_transform(X_train)
X_test      = scaler.transform(X_test)


# ─────── Activation Functions ───────
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ─────── Parameter Initialization ───────
def initialize_parameters():
    np.random.seed(0)
    params = {
        # layer 1: 4 inputs → 8 neurons
        'W1': np.random.randn(8, 4) * np.sqrt(2 / 4),
        'b1': np.zeros((8, 1)),
        # layer 2: 8 → 6
        'W2': np.random.randn(6, 8) * np.sqrt(2 / 8),
        'b2': np.zeros((6, 1)),
        # layer 3: 6 → 3
        'W3': np.random.randn(3, 6) * np.sqrt(2 / 6),
        'b3': np.zeros((3, 1)),
        # output layer: 3 → 3
        'W4': np.random.randn(3, 3) * np.sqrt(2 / 3),
        'b4': np.zeros((3, 1))
    }
    return params



# ─────── Forward Pass ───────
def forward_pass(X, params):
    # layer 1
    Z1 = X @ params['W1'].T + params['b1'].T
    A1 = relu(Z1)

    # layer 2
    Z2 = A1 @ params['W2'].T + params['b2'].T
    A2 = relu(Z2)

    # layer 3
    Z3 = A2 @ params['W3'].T + params['b3'].T
    A3 = relu(Z3)

    # output layer (layer 4)
    Z4 = A3 @ params['W4'].T + params['b4'].T
    A4 = softmax(Z4)

    cache = {
        'A0': X,
        'Z1': Z1, 'A1': A1,
        'Z2': Z2, 'A2': A2,
        'Z3': Z3, 'A3': A3,
        'Z4': Z4, 'A4': A4
    }
    return A4, cache


# ─────── Loss ───────
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    # pick the log‐prob of the correct class
    log_probs = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-9)
    return np.sum(log_probs) / m


# ─────── Backward Pass ───────
def backward_pass(y_true, params, cache):
    m     = y_true.shape[0]
    grads = {}

    # output layer (softmax + CE)
    dZ4        = (cache['A4'] - y_true) / m
    grads['W4'] = dZ4.T @ cache['A3']                            # (3,3)
    grads['b4'] = np.sum(dZ4, axis=0, keepdims=True).T           # (3,1)

    # layer 3 (ReLU)
    dA3        = dZ4 @ params['W4']                              # (m,3)
    dZ3        = dA3 * relu_derivative(cache['Z3'])
    grads['W3'] = dZ3.T @ cache['A2']                            # (3,6)
    grads['b3'] = np.sum(dZ3, axis=0, keepdims=True).T           # (3,1)

    # layer 2 (ReLU)
    dA2        = dZ3 @ params['W3']                              # (m,6)
    dZ2        = dA2 * relu_derivative(cache['Z2'])
    grads['W2'] = dZ2.T @ cache['A1']                            # (6,8)
    grads['b2'] = np.sum(dZ2, axis=0, keepdims=True).T           # (6,1)

    # layer 1 (ReLU)
    dA1        = dZ2 @ params['W2']                              # (m,8)
    dZ1        = dA1 * relu_derivative(cache['Z1'])
    grads['W1'] = dZ1.T @ cache['A0']                            # (8,4)
    grads['b1'] = np.sum(dZ1, axis=0, keepdims=True).T           # (8,1)

    return grads


# ─────── Parameter Update ───────
def update_parameters(params, grads, lr):
    # this will now pick up W1–W4 and b1–b4 automatically
    for key in params:
        params[key] -= lr * grads[key]
    return params



# ─────── Training Loop ───────
def train(X_train, y_train, epochs=2000, lr=0.01, print_every=200):
    params = initialize_parameters()
    for epoch in range(1, epochs+1):
        # forward + loss
        y_pred, cache = forward_pass(X_train, params)
        loss = compute_loss(y_train, y_pred)

        # backward + update
        grads = backward_pass(y_train, params, cache)
        params = update_parameters(params, grads, lr)

        if epoch % print_every == 0:
            print(f"Epoch {epoch:4d}  Loss: {loss:.4f}")
    return params


# ─────── Evaluation ───────
def evaluate(X, y_true, params):
    y_pred, _   = forward_pass(X, params)
    preds       = np.argmax(y_pred, axis=1)
    labels      = np.argmax(y_true, axis=1)
    accuracy    = np.mean(preds == labels)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%\n")

    # detailed breakdown
    from collections import Counter
    print("Predicted class distribution:", Counter(preds))
    names = load_iris().target_names
    print("\nSample predictions (first 10):")
    for i in range(10):
        print(f" {i+1:2d}: Predicted={names[preds[i]]:<10s}  Actual={names[labels[i]]}")


# ─────── Main ───────
if __name__ == "__main__":
    params = train(X_train, y_train, epochs=2000, lr=0.01, print_every=200)
    evaluate(X_test, y_test, params)
