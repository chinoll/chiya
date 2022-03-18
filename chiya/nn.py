from .autograd import Tensor,Dependency
import numpy as np
from typing import Iterator, Any, List
import inspect
from numba import njit

class Parameter(Tensor):
    pass

class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def eval(self):
        self.training = False
        for parameter in self.parameters():
            parameter.requires_grad = False

    def train(self):
        self.training = True
        for parameter in self.parameters():
            parameter.requires_grad = True

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

class Linear(Module):
    def __init__(self,out_features,in_features=None,bias=True):
        super().__init__()
        if in_features == None:
            self.weight = None
            self.out_features = out_features
        else:
            self.weight = Parameter(normal((in_features, out_features)),requires_grad=True)
        if bias:
            self.b = Parameter(np.zeros(out_features),requires_grad=True)
        else:
            self.b = None

    def forward(self, inputs):
        depends_on: List[Dependency] = []
        if self.weight is None:
            in_featues = np.prod(inputs.shape[1:])
            self.weight = Parameter(normal((in_featues, self.out_features)),requires_grad=True)
        out = np.dot(inputs.data,self.weight.data)
        if self.b is not None:
            out += self.b.data
        if inputs.requires_grad:
            def LinearGradNative(grad:np.ndarray):
                return grad @ self.weight.data.T
            depends_on.append(Dependency(inputs,LinearGradNative))
        if self.weight.requires_grad:
            def LinearWeightGradNative(grad:np.ndarray):
                return inputs.data.T @ grad
            depends_on.append(Dependency(self.weight,LinearWeightGradNative))
        if self.b.requires_grad:
            def LinearBiasGradNative(grad:np.ndarray):
                return np.sum(grad, axis=0)
            depends_on.append(Dependency(self.b,LinearBiasGradNative))
        requires_grad = inputs.requires_grad or self.weight.requires_grad or self.b.requires_grad
        return Tensor(out,requires_grad=requires_grad,depends_on=depends_on)

class ReLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        depends_on: List[Dependency] = []
        if inputs.requires_grad:
            def BackwardReLU(grad:np.ndarray):
                return np.where(inputs.data>0,grad,0)
            depends_on.append(Dependency(inputs,BackwardReLU))
        return Tensor(np.maximum(0,inputs.data),inputs.requires_grad,depends_on=depends_on)

class CrossEntropy(Module):
    def __init__(self):
        super().__init__()

    def forward(self,y_pred,y):

        eps = np.finfo(float).eps
        cross_entropy = -np.sum(y.data * np.log(y_pred.data + eps)) / y.shape[0]

        depends_on: List[Dependency] = []
        if y_pred.requires_grad:
            def CrossEntropyGradNative(grad:np.ndarray):
                return (y_pred.data - y.data)
            depends_on.append(Dependency(y_pred,CrossEntropyGradNative))
        return Tensor(cross_entropy,requires_grad=y_pred.requires_grad,depends_on=depends_on)

class Softmax(Module):
    def __init__(self, dim=-1,cross_entropy=False):
        super().__init__()

        self.dim = dim
        self.n_in = None
        self.cross_entropy = cross_entropy

    def forward(self, X):
        ex = np.exp(X.data - np.max(X.data, axis=self.dim, keepdims=True))
        Y = ex / np.sum(ex,axis=self.dim,keepdims=True)

        depends_on: List[Dependency] = []
        if X.requires_grad:
            def BackwardSoftmax(grad:np.ndarray):
                if self.cross_entropy:
                    return grad
                else:
                    dX = []
                    nonlocal X
                    for dy, x in zip([grad], [X.data]):
                        dx = self._bwd(dy, x)
                        dX.append(dx)

                    dout = dX[0] if len(X.data) == 1 else np.array(dX).squeeze(0)
                    return dout
            depends_on.append(Dependency(X,BackwardSoftmax))

        return Tensor(Y,X.requires_grad,depends_on=depends_on)

    def _bwd(self, dLdy, X):
        dX = []
        ex = lambda X:np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        f = lambda X: ex(X)/ np.sum(X,axis=self.dim,keepdims=True)
        for dy, x in zip(dLdy, X):
            dxi = []
            for dyi, xi in zip(*np.atleast_2d(dy, x)):
                yi = f(xi.reshape(1, -1)).reshape(-1, 1)
                dyidxi = np.diagflat(yi) - yi @ yi.T  # jacobian wrt. input sample xi
                dxi.append(dyi @ dyidxi)
            dX.append(dxi)
        return np.array(dX).reshape(*X.shape)

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

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()

@njit
def _col2im(col, input_shape, filter_h, filter_w, stride, out_h, out_w):
    N, C, H, W = input_shape
    tmp1 = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    tmp2 = np.transpose(tmp1, axes=(0, 3, 4, 5, 1, 2))
    img = np.zeros((N, C, H + stride - 1, W + stride - 1)).astype('float32')
    for i in range(filter_h):
        i_max = i + stride*out_h
        for j in range(filter_w):
            j_max = j + stride*out_w
            img[:, :, i:i_max:stride, j:j_max:stride] += tmp2[:, :, i, j, :, :]
    return img[:, :, 0:H, 0:W]

@njit
def _im2col(img, kh,kw, stride=1):
    #b c h w -> b h w c
    img = img.copy().transpose(0, 2, 3, 1)
    N, H, W, C = img.shape

    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    col = np.empty((N * out_h * out_w, kh * kw * C),dtype=img.dtype)
    outsize = out_w * out_h

    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + kh
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + kw
            col[y_start+x::outsize, :] = img[:, y_min:y_max, x_min:x_max, :].copy().reshape(N, -1)
    return col

class Conv2d(Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True):
        super().__init__()
        self.kernel = Parameter(normal((out_channels,in_channels,kernel_size,kernel_size)).astype('float32'),requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.bias = Parameter(np.zeros((1,out_channels)).astype('float32'),requires_grad=bias)
    def forward(self, x):
        outc,inc,kh,kw = self.kernel.shape
        stride = self.stride
        X = x.data
        N, _, in_h, in_w = X.shape
        out_h = (in_h - kh) // stride + 1
        out_w = (in_w - kw) // stride + 1
        depends_on: List[Dependency] = []

        col = _im2col(X, kh, kw, stride)
        W = self.kernel.data.reshape(outc,-1).T
        B = self.bias.data.reshape(-1,outc)
        out = col @ W + B
        out = out.reshape(N, out_h, out_w, outc).transpose(0, 3, 1, 2)

        if self.kernel.requires_grad:
            def BackwardConv2dKernel(grad:np.ndarray):
                grad = np.transpose(grad, (0, 2, 3, 1)).reshape(-1,outc) / N
                col_dW = np.dot(col.T, grad)
                dW = np.transpose(col_dW, axes=(1, 0)).reshape(outc, inc, kh, kw)
                return dW
            depends_on.append(Dependency(self.kernel,BackwardConv2dKernel))

        if self.bias.requires_grad:
            def BackwardConv2dBias(grad:np.ndarray):
                grad = np.transpose(grad, (0, 2, 3, 1)).reshape(-1,outc)
                col_dB = np.sum(grad,axis=0,keepdims=True) / N
                return col_dB
            depends_on.append(Dependency(self.bias,BackwardConv2dBias))

        if x.requires_grad:
            def BackwardConv2dX(grad:np.ndarray):
                grad = grad.transpose(0, 2, 3, 1).reshape(-1,outc)
                dx = grad @ W.T
                d_in = _col2im(dx, X.shape, kh, kw, stride, out_h, out_w)
                return d_in
            depends_on.append(Dependency(x,BackwardConv2dX))

        return Tensor(out,self.kernel.requires_grad,depends_on)

class flatten(Module):
    def forward(self,x):
        return x.reshape(x.shape[0],-1)

def _maxpool2d(x, ksize, stride=1):
    N, C, H, W = x.shape
    out_h = (H - ksize[0]) // stride + 1
    out_w = (W - ksize[1]) // stride + 1
    col = _im2col(x.data, ksize[0],ksize[1], stride)
    col_x = col.reshape(-1, np.prod(ksize))
    arg_max = np.argmax(col_x, axis=1)
    out1 = np.max(col_x, axis=1)
    out2 = out1.reshape(N, out_h, out_w, C)
    out = np.transpose(out2, axes=(0,3,1,2))
    requires_grad = x.requires_grad
    depends_on: List[Dependency] = []

    if requires_grad:
        def BackwardMaxPool2d(grad:np.ndarray):
            dout = np.transpose(grad, (0,2,3,1))
            dmax = np.zeros((dout.size, ksize[0]*ksize[1])).astype('float32')
            dmax[np.arange(arg_max.size), arg_max.flatten()] = dout.flatten()
            dmax = dmax.reshape(dout.shape + (np.prod(ksize),))
            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            dx = _col2im(dcol, x.shape, ksize[0], ksize[1], stride, out_h, out_w)
            return dx
        depends_on.append(Dependency(x, BackwardMaxPool2d))
    return Tensor(out,requires_grad,depends_on)

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0, padding_mode='zero'):
        super().__init__()
        self.kernel_size = (kernel_size,kernel_size) if type(kernel_size) == int else kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, x):
        return _maxpool2d(x,self.kernel_size,self.stride)


def _avgpool2d(x, ksize, stride=1):
    N, C, H, W = x.shape
    out_h = (H - ksize[0]) // stride + 1
    out_w = (W - ksize[1]) // stride + 1
    col = _im2col(x.data, ksize[0],ksize[1], stride)
    col_x = col.reshape(-1, np.prod(ksize))
    out = np.mean(col_x, axis=1)
    out_size = out.size
    out = out.reshape(N, out_h, out_w, C)
    out = np.transpose(out, axes=(0,3,1,2))

    requires_grad = x.requires_grad
    depends_on: List[Dependency] = []

    if requires_grad:
        def BackwardAvgPool2d(grad:np.ndarray):
            grad = np.transpose(grad, (0,2,3,1))
            dmean = np.zeros((grad.size, np.prod(ksize))).astype(np.float32)
            dmean[np.arange(out_size), 1] = grad.flatten()/np.prod(ksize)
            dmean = dmean.reshape(grad.shape + (np.prod(ksize),))
            dcol = dmean.reshape(dmean.shape[0] * dmean.shape[1] * dmean.shape[2], -1)
            dx = _col2im(dcol, x.shape, ksize[0], ksize[1], stride, out_h, out_w)
            return dx
        depends_on.append(Dependency(x, BackwardAvgPool2d))
    return Tensor(out,requires_grad,depends_on)

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0, padding_mode='zero'):
        super().__init__()
        self.kernel_size = (kernel_size,kernel_size) if type(kernel_size) == int else kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, x):
        return _avgpool2d(x,self.kernel_size,self.stride)

class BatchNorm2d(Module):
    def __init__(self,num_features,eps=1e-5,momentum=0.99):
        super(BatchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros((1,num_features,1,1),dtype=np.float32)
        self.running_var = np.zeros((1,num_features,1,1),dtype=np.float32)
        self.beta = Parameter(np.zeros((1,num_features,1,1),dtype=np.float32),requires_grad=True)
        self.gamma = Parameter(np.ones((1,num_features,1,1),dtype=np.float32),requires_grad=True)
    def forward(self,x):
        N, _, _, _ = x.shape
        if self.training:
            mu = np.mean(x.data, axis = 0)

            xmu = x.data - mu

            var = np.var(xmu ** 2, axis = 0) + self.eps

            std = np.sqrt(var)

            xhat = xmu / std

            out = self.gamma.data * xhat + self.beta.data

            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
        else:
            print("Not in training mode")
            mu = self.running_mean
            var = self.running_var
            norm_x = (x.data - mu) / np.sqrt(var + self.eps)
            out = self.gamma.data * norm_x + self.beta.data

        depends_on: List[Dependency] = []
        if x.requires_grad:
            def BackwardBatchNorm2dX(grad:np.ndarray):
                dxhat = grad * self.gamma.data

                divar = np.sum(dxhat*xmu, axis=0)
                dxmu1 = dxhat / var

                dsqrtvar = -1. / var * divar

                dvar = 0.5 * 1. /np.sqrt(var+self.eps) * dsqrtvar

                dsq = dvar / N

                dxmu2 = 2 * xmu * dsq

                dx1 = (dxmu1 + dxmu2)
                dmu = -np.sum(dxmu1+dxmu2, axis=0)

                dx2 = dmu / N

                dx = dx1 + dx2
                return dx
            def BackwardBatchNorm2dBeta(grad:np.ndarray):
                return np.sum(grad, axis=0)
            def BackwardBatchNorm2dGamma(grad:np.ndarray):
                return np.sum(grad*xhat, axis=0)
            depends_on.append(Dependency(x,BackwardBatchNorm2dX))
            depends_on.append(Dependency(self.gamma,BackwardBatchNorm2dGamma))
            depends_on.append(Dependency(self.beta,BackwardBatchNorm2dBeta))

        return Tensor(out,requires_grad=True,depends_on=depends_on)

def normal(shape):
    return np.random.normal(size=shape)*0.001