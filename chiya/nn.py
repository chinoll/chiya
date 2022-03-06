from turtle import forward

from matplotlib.pyplot import axes
from .autograd import Tensor,Dependency
import numpy as np
from typing import Iterator, Any, List
import inspect
from numba import jit

class Parameter(Tensor):
    pass


class Module:
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

class linear(Module):
    def __init__(self, input_size, output_size):
        self.W = Parameter(np.random.randn(input_size, output_size)*0.001,requires_grad=True)
        self.b = Parameter(np.zeros(output_size),requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.b

class sigmod(Module):
    def forward(self, x):
        return 1 / (1 + x.exp())

class softmax(Module):
    def forward(self, x):
        x = x - x.max(axis=1,keepdims=True)
        e = x.exp()
        return e / e.sum(axis=1,keepdims=True)

class ReLU(Module):
    def forward(self, x):
        return x.maximum(0)

class cross_entropy(Module):
    def __init__(self, one_hot=False, softmax_out=True):
        self.one_hot = one_hot
        self.softmax_out = softmax_out

    def forward(self, x, y):
        return -(y * (x+1e-8).log()).sum()/x.data.shape[0]

class Sequential(Module):
    def __init__(self, *layers, lr=0.01):
        self.lr = lr
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self) -> Iterator[Parameter]:
        for i in self.layers:
            yield from i.parameters()
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

@jit(nopython=True)
def _col2im(col, input_shape, filter_h, filter_w, stride, pad, out_h, out_w):
    N, C, H, W = input_shape
    tmp1 = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    tmp2 = np.transpose(tmp1, axes=(0, 3, 4, 5, 1, 2))
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for i in range(filter_h):
        i_max = i + stride*out_h
        for j in range(filter_w):
            j_max = j + stride*out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += tmp2[:, :, i, j, :, :]
    return img[:, :, pad:H + pad, pad:W + pad]

@jit(nopython=True)
def _im2col(img, ksize, stride=1):
    #b c h w -> b h w c
    img = img.copy().transpose(0, 2, 3, 1)
    N, H, W, C = img.shape

    out_h = (H - ksize[0]) // stride + 1
    out_w = (W - ksize[1]) // stride + 1
    col = np.empty((N * out_h * out_w, ksize[0] * ksize[1] * C),dtype=img.dtype)
    outsize = out_w * out_h

    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + ksize[0]
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + ksize[1]
            col[y_start+x::outsize, :] = img[:, y_min:y_max, x_min:x_max, :].copy().reshape(N, -1)
    return col

def _conv2d(x, W, stride=1, padding=0):
    N, _, H, w = x.shape
    F, C, kH, kW = W.shape
    out_h = (H - kH) // stride + 1
    out_w = (w - kW) // stride + 1
    col_x = _im2col(x.data, (kH, kW), stride)
    col_W = W.data.reshape(F, -1).T
    out = col_x @ col_W
    requires_grad = x.requires_grad or W.requires_grad
    depends_on: List[Dependency] = []

    if W.requires_grad:
        def grad_fn1(grad:np.ndarray):
            delta_in_2d = np.transpose(grad, (0, 2, 3, 1)).reshape(-1,F)
            dw = np.dot(col_x.T, delta_in_2d).transpose(1, 0)
            dw = dw.reshape(F, C, kH, kW) / N
            return dw
        depends_on.append(Dependency(W, grad_fn1))

    if x.requires_grad:
        def grad_fn3(grad:np.ndarray):
            delta_in_2d = np.transpose(grad, (0, 2, 3, 1)).reshape(-1,F)
            dcol = np.dot(delta_in_2d,col_W.T)
            delta_out = _col2im(dcol, x.shape, kH, kW, stride,padding,out_h,out_w)
            return delta_out
        depends_on.append(Dependency(x, grad_fn3))

    out = out.reshape(N,out_h,out_w,F)
    return Tensor(np.transpose(out,axes=(0,3,1,2)), requires_grad, depends_on)

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,padding_mode='zeros'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.kernel = Parameter(np.random.randn(out_channels, in_channels, kernel_size, kernel_size)*0.01,requires_grad=True)
        self.bias = Parameter(np.zeros((1,out_channels,1,1)),requires_grad=True)

    def forward(self, x):
        return _conv2d(x, self.kernel, self.stride, self.padding) + self.bias

class flatten(Module):
    def forward(self, x):
        return x.flatten()