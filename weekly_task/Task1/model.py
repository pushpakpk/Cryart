import numpy as np

# ----------------------------
# Dense Layer
# ----------------------------
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ, learning_rate):
        m = self.X.shape[0]
        dW = (self.X.T @ dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T

        # Update weights
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        return dX


# ----------------------------
# Activations
# ----------------------------
class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        return dA * (self.Z > 0)


# ----------------------------
# Loss Function (MSE)
# ----------------------------
class MSELoss:
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]
