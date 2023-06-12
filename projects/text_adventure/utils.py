import numpy as np 

# activation functions

def linear(X):
    return X

def linear_prime(X):
    return np.ones_like(X)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    # helps prevent vanishing gradients
    return np.maximum(x, 0)

def relu_prime(x):
    # wrong for x=0 but this seems to be convention
    return np.where(x >= 0, 1.0, 0.0)

def elu(x, alpha=1.0):
    #x = np.clip(x,-1,1)
    ex_x_alpha = alpha * (np.exp(x)- 1)
    return np.where(x >= 0, x, ex_x_alpha)


def elu_prime(x, alpha=1.0):
    x_alpha = elu(x) + alpha
    return np.where(x >= 0, 1, x_alpha)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_prime(x):
    softmax_output = softmax(x)
    return softmax_output * (1 - softmax_output)


def relu_init(W: np.ndarray) -> np.ndarray:
    """
    Apply He initialization to the random weight matrix. 
    Used when the chosen non-linearity is ReLU.
    """
    denom = 1
    if W.shape[0] > 1:
        denom = (W.shape[0] - 1)
    return W * np.sqrt(2/(denom))


# loss functions

def logloss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))

def logloss_prime(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
    
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)
    
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(y_true-y_pred)

def huber(y_true, y_pred, delta=1):
    error = y_true - y_pred
    quadratic_term = np.minimum(np.abs(error), delta)
    linear_term = np.abs(error) - quadratic_term
    loss = 0.5 * quadratic_term**2 + delta * linear_term
    return np.mean(loss)

def huber_prime(y_true, y_pred, delta=1):
    error = y_true - y_pred
    condition = np.abs(error) <= delta
    gradient = np.where(condition, error, delta * np.sign(error))
    return gradient



