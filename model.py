"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
from numpy.typing import NDArray


def argmax_rows(matrix: NDArray) -> NDArray:
    '''return the index of the largest element in each row of a 2D array'''

    return matrix.argmax(axis=-1)

# Step 2 - row_max
import numpy as np
from numpy.typing import NDArray


def row_max(matrix: NDArray) -> NDArray:
    '''return the maximum value of each row of `matrix` with keepdims True for broadcasting.'''

    return matrix.max(axis=-1, keepdims=True)

# Step 3 - row_sum
import numpy as np
from numpy.typing import NDArray


def row_sum(matrix: NDArray) -> NDArray:
    """Return per-row sums of a 2D array with shape (N, 1)."""

    return matrix.sum(axis=-1, keepdims=True)

# Step 4 - exp_shifted
import numpy as np
from numpy.typing import NDArray


def exp_shifted(logits: NDArray) -> NDArray:
    """Subtract per-row max from logits and exponentiate elementwise."""

    return np.exp(logits - row_max(logits))

# Step 5 - stable_softmax
import numpy as np
from numpy.typing import NDArray


def stable_softmax(logits: NDArray):
    '''Compute a numerically stable softmax row-wise over (N, C) logits.'''

    exp_logits = exp_shifted(logits)
    return exp_logits / row_sum(exp_logits)

# Step 6 - one_hot
import numpy as np
from numpy.typing import NDArray


def one_hot(labels: NDArray, num_classes: int):
    '''convert integer labels into a (N, num_classes) one-hot float matrix'''

    matrix = np.zeros((labels.size, num_classes))
    matrix[np.arange(labels.size), labels] = 1
    return matrix

# Step 7 - gather_true_class_probs
import numpy as np
from numpy.typing import NDArray


def gather_true_class_probs(probs: NDArray, labels: NDArray):
    '''return probs[i, labels[i]] for every row i as a 1D length-N array.'''

    return probs[np.arange(labels.size), labels]

# Step 8 - cross_entropy_loss
import numpy as np
from numpy.typing import NDArray


def cross_entropy_loss(probs: NDArray, labels: NDArray, eps: float = 1e-12):
    '''return the mean negative log-likelihood of the true-class probabilities'''

    return -np.mean(np.log(gather_true_class_probs(probs, labels).clip(eps, 1.0)))

# Step 9 - accuracy
import numpy as np
from numpy.typing import NDArray


def accuracy(logits_or_probs: NDArray, labels: NDArray) -> float:
    '''return the fraction of rows whose argmax matches the integer label.'''

    return np.mean((logits_or_probs.argmax(axis=-1) == labels))

# Step 10 - he_std
import numpy as np


def he_std(fan_in: int) -> float:
    '''return the He initialization standard deviation sqrt(2 / fan_in).'''

    return np.sqrt(2 / fan_in)

# Step 11 - he_init
import numpy as np


def he_init(shape: tuple[int, ...], fan_in: int, seed=None):
    '''sample a weight tensor from a normal distribution scaled by He std using the seed.'''

    np.random.seed(seed)
    return np.random.normal(0, he_std(fan_in), shape)

# Step 12 - init_zero_bias
import numpy as np


def init_zero_bias(length: int):
    '''return a 1D float array of zeros with the given length.'''

    return np.zeros(length)

# Step 13 - pad_2d
import numpy as np
from numpy.typing import NDArray


def pad_2d(images: NDArray, pad: int):
    '''zero-pad the spatial (H, W) dims of a 4D (N, C, H, W) tensor by `pad` on each side.'''

    return np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

# Step 14 - output_spatial_size
def output_spatial_size(input_size: int, kernel: int, stride: int, padding: int):
    '''return the conv/pool output spatial dimension from input_size, kernel, stride, padding'''

    return (input_size + 2 * padding - kernel) // stride + 1

# Step 15 - im2col
import numpy as np
from numpy.typing import NDArray


def im2col(images: NDArray, kernel_h: int, kernel_w: int, stride: int, padding: int):
    '''Unroll overlapping patches of a 4D image tensor into a 2D column matrix.'''

    N, C, H, W = images.shape

    padded = pad_2d(images, padding)

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    cols = np.empty((N, out_h, out_w, C, kernel_h, kernel_w), dtype=images.dtype)
    for i in range(kernel_h):
        h_slice = slice(i, i + out_h * stride, stride)
        for j in range(kernel_w):
            w_slice = slice(j, j + out_w * stride, stride)
            cols[..., i, j] = padded[..., h_slice, w_slice].transpose(0, 2, 3, 1)

    return cols.reshape(N * out_h * out_w, C * kernel_h * kernel_w)

# Step 16 - col2im
import numpy as np
from numpy.typing import NDArray


def col2im(
    cols: NDArray,
    input_shape: tuple[int, ...],
    kernel_h: int,
    kernel_w: int,
    stride: int,
    padding: int,
):
    '''re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor'''

    N, C, H, W = input_shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    cols = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w)

    padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)
    for i in range(kernel_h):
        h_slice = slice(i, i + out_h * stride, stride)
        for j in range(kernel_w):
            w_slice = slice(j, j + out_w * stride, stride)
            padded[..., h_slice, w_slice] += cols[..., i, j].transpose(0, 3, 1, 2)

    images = padded[..., padding:-padding, padding:-padding] if padding else padded
    return images

# Step 17 - conv2d_forward
from numpy.typing import NDArray


def conv2d_forward(x: NDArray, weights: NDArray, bias: NDArray, stride: int, padding: int):
    '''convolve x with weights using im2col, add bias, return output and a backprop cache.'''

    N, C_in, H, W = x.shape
    C_out, weight_c_in, kernel_h, kernel_w = weights.shape

    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)

    cols = im2col(x, kernel_h, kernel_w, stride, padding)
    out_cols = cols @ weights.reshape(C_out, -1).T + bias
    out = out_cols.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
    return out, {
        'x_shape': x.shape,
        'weights': weights,
        'cols': cols,
        'stride': stride,
        'padding': padding,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
    }

# Step 18 - conv2d_grad_input
from numpy.typing import NDArray


def conv2d_grad_input(d_out: NDArray, cache: dict[str, tuple[int, ...] | NDArray | int]):
    '''backprop d_out through the conv input using col2im'''

    weights: NDArray = cache['weights']

    C_out = weights.shape[0]

    d_out_row = d_out.transpose(0, 2, 3, 1).reshape(-1, C_out)
    W_row = weights.reshape(C_out, -1)
    d_cols = d_out_row @ W_row
    dx = col2im(
        d_cols,
        cache['x_shape'],
        cache['kernel_h'],
        cache['kernel_w'],
        cache['stride'],
        cache['padding'],
    )
    return dx

# Step 19 - conv2d_grad_weights
from numpy.typing import NDArray


def conv2d_grad_weights(d_out: NDArray, cache: dict[str, tuple[int, ...] | NDArray | int]):
    '''return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache.'''

    weights: NDArray = cache['weights']
    cols: NDArray = cache['cols']
    kernel_h: int = cache['kernel_h']
    kernel_w: int = cache['kernel_w']

    C_out, C_in, *_ = weights.shape

    d_W_row = d_out.transpose(1, 0, 2, 3).reshape(C_out, -1) @ cols
    d_W = d_W_row.reshape(C_out, C_in, kernel_h, kernel_w)
    return d_W

# Step 20 - conv2d_grad_bias
from numpy.typing import NDArray


def conv2d_grad_bias(d_out: NDArray) -> NDArray:
    '''return a length C_out gradient by reducing d_out over batch and spatial axes'''

    return d_out.sum(axis=(0, 2, 3))

# Step 21 - conv2d_backward
from numpy.typing import NDArray


def conv2d_backward(d_out: NDArray, cache: dict[str, tuple[int, ...] | NDArray | int]):
    '''return (dx, dW, db) using the conv2d gradient helpers and the forward cache'''

    return (
        conv2d_grad_input(d_out, cache),
        conv2d_grad_weights(d_out, cache),
        conv2d_grad_bias(d_out),
    )

# Step 22 - maxpool2d_forward
import numpy as np
from numpy.typing import NDArray


def maxpool2d_forward(x: NDArray, kernel: int, stride: int):
    '''run 2D max pooling and cache the in-window argmax of each output cell.'''

    N, C, H, W = x.shape

    out_h = (H - kernel) // stride + 1
    out_w = (W - kernel) // stride + 1

    out = np.empty((N, C, out_h, out_w), dtype=x.dtype)
    argmax = np.empty((N, C, out_h, out_w), dtype=np.int64)
    for i in range(out_h):
        h_slice = slice(i * stride, i * stride + kernel)
        for j in range(out_w):
            w_slice = slice(j * stride, j * stride + kernel)
            window = x[..., h_slice, w_slice]
            out[..., i, j] = np.max(window, axis=(-1, -2))
            argmax[..., i, j] = np.argmax(window.reshape(N, C, -1), axis=-1)

    return out, {'x_shape': x.shape, 'argmax': argmax, 'kernel': kernel, 'stride': stride}

# Step 23 - scatter_grad_window
import numpy as np


def scatter_grad_window(grad_value: float, argmax_index: int, kernel: int):
    '''place grad_value at the argmax position within a (kernel, kernel) zero array.'''

    window = np.zeros(kernel * kernel)
    window[argmax_index] = grad_value
    return window.reshape(kernel, kernel)

# Step 24 - maxpool2d_backward
import numpy as np
from numpy.typing import NDArray


def maxpool2d_backward(d_out: NDArray, cache: dict[str, tuple[int, ...] | NDArray | int]):
    '''scatter each d_out value to the cached argmax position in its window'''

    x_shape: tuple[int, ...] = cache['x_shape']
    argmax: NDArray = cache['argmax']
    kernel: int = cache['kernel']
    stride: int = cache['stride']

    N, C, out_h, out_w = argmax.shape

    d_maxpool2d = np.zeros(x_shape, dtype=d_out.dtype)
    for i in range(out_h):
        h_slice = slice(i * stride, i * stride + kernel)
        for j in range(out_w):
            w_slice = slice(j * stride, j * stride + kernel)
            d_maxpool2d[..., h_slice, w_slice] += scatter_grad_window(
                d_out[..., i, j], argmax[..., i, j], kernel
            )

    return d_maxpool2d

# Step 25 - relu_forward
import numpy as np
from numpy.typing import NDArray


def relu_forward(x: NDArray):
    '''Compute the elementwise ReLU and cache the input for backprop.'''

    return np.maximum(x, 0), {'x': x}

# Step 26 - relu_backward
import numpy as np
from numpy.typing import NDArray


def relu_backward(d_out: NDArray, cache: dict[str, NDArray]):
    '''mask the upstream gradient by the positive entries of the cached input.'''

    return np.where(cache['x'] > 0, d_out, 0)

# Step 27 - flatten_forward
from numpy.typing import NDArray


def flatten_forward(x: NDArray):
    '''reshape a 4D feature map into a 2D batch matrix and cache the original shape'''

    return x.reshape(x.shape[0], -1), {'x_shape': x.shape}

# Step 28 - flatten_backward
import numpy as np
from numpy.typing import NDArray


def flatten_backward(d_out: NDArray, cache: dict[str, tuple[int, ...]]):
    '''reshape the upstream gradient back to the original 4D feature map shape.'''
    
    return d_out.reshape(cache['x_shape'])

# Step 29 - linear_forward
from numpy.typing import NDArray


def linear_forward(x: NDArray, weights: NDArray, bias: NDArray):
    '''compute X @ W + b and cache the inputs needed for backprop.'''

    return x @ weights + bias, {'x': x, 'weights': weights}

# Step 30 - linear_grad_input
import numpy as np
from numpy.typing import NDArray


def linear_grad_input(d_out: NDArray, cache: dict[str, NDArray]):
    """Gradient of a linear layer w.r.t. its input X."""
    
    return d_out @ cache['weights'].T

# Step 31 - linear_grad_weights
import numpy as np
from numpy.typing import NDArray


def linear_grad_weights(x: NDArray, dout: NDArray):
    """Gradient of loss wrt linear-layer weights W of shape (D_in, D_out)."""

    return x.T @ dout

# Step 32 - linear_grad_bias
import numpy as np
from numpy.typing import NDArray


def linear_grad_bias(dout: NDArray) -> NDArray:
    '''Compute the bias gradient of a linear layer given upstream gradient dout.'''

    return dout.sum(axis=0)

# Step 33 - linear_backward
from numpy.typing import NDArray


def linear_backward(dout: NDArray, cache: dict[str, NDArray]):
    '''combine input, weight, and bias gradients for a linear layer using the cache'''

    return (
        linear_grad_input(dout, cache),
        linear_grad_weights(cache['x'], dout),
        linear_grad_bias(dout),
    )

# Step 34 - softmax_cross_entropy_forward
from numpy.typing import NDArray


def softmax_cross_entropy_forward(logits: NDArray, y: NDArray):
    '''return the mean cross-entropy loss for logits (N, C) and integer labels y (N,).'''

    return cross_entropy_loss(stable_softmax(logits), y) + 0

# Step 35 - softmax_cross_entropy_backward
import numpy as np
from numpy.typing import NDArray


def softmax_cross_entropy_backward(logits: NDArray, y: NDArray) -> NDArray:
    '''return the fused softmax-cross-entropy gradient of shape (N, C).'''

    N, C = logits.shape

    dlogits = stable_softmax(logits)
    dlogits[np.arange(N), y] -= 1.0
    dlogits /= N

    # 理论上所有元素之和为 0，修正浮点计算产生的微小残差
    dlogits.flat[-1] -= dlogits.sum()

    return dlogits

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

