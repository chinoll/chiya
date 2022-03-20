from .autograd import Tensor,Dependency
import numpy as np
from typing import Iterator, Any, List
import inspect
from numba import njit

# @njit
def _col2im(col, input_shape, filter_h, filter_w, stride, out_h, out_w):
    N, C, H, W = input_shape
    tmp1 = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    tmp2 = np.transpose(tmp1, axes=(0, 3, 4, 5, 1, 2))
    img = np.zeros((N, C, H + stride[0] - 1, W + stride[1] - 1)).astype('float32')
    for i in range(filter_h):
        i_max = i + stride[0] * out_h
        for j in range(filter_w):
            j_max = j + stride[1] * out_w
            img[:, :, i:i_max:stride[0], j:j_max:stride[1]] += tmp2[:, :, i, j, :, :]
    return img[:, :, 0:H, 0:W]

@njit
def _im2col(img, kh, kw, stride):
    #b c h w -> b h w c
    img = img.copy().transpose(0, 2, 3, 1)
    N, H, W, C = img.shape

    out_h = (H - kh) // stride[0] + 1
    out_w = (W - kw) // stride[1] + 1
    col = np.empty((N * out_h * out_w, kh * kw * C),dtype=img.dtype)
    outsize = out_w * out_h

    for y in range(out_h):
        y_min = y * stride[0]
        y_max = y_min + kh
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride[1]
            x_max = x_min + kw
            col[y_start+x::outsize, :] = img[:, y_min:y_max, x_min:x_max, :].copy().reshape(N, -1)
    return col

def MaxPool2d(x, ksize, stride):
    N, C, H, W = x.shape
    out_h = (H - ksize[0]) // stride[0] + 1
    out_w = (W - ksize[1]) // stride[1] + 1
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

def AvgPool2d(x, ksize, stride):
    N, C, H, W = x.shape
    out_h = (H - ksize[0]) // stride[0] + 1
    out_w = (W - ksize[1]) // stride[1] + 1
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

def Conv2d(x,kernel,bias,stride):
    outc,inc,kh,kw = kernel.shape
    X = x.data
    N, _, in_h, in_w = X.shape

    out_h = (in_h - kh) // stride[0] + 1
    out_w = (in_w - kw) // stride[1] + 1

    depends_on: List[Dependency] = []

    col = _im2col(X, kh, kw, stride)
    W = kernel.data.reshape(outc,-1).T
    out = col @ W

    if bias:
        B = bias.data.reshape(-1,outc)
        out += B

    out = out.reshape(N, out_h, out_w, outc).transpose(0, 3, 1, 2)

    if kernel.requires_grad:
        def BackwardConv2dKernel(grad:np.ndarray):
            grad = np.transpose(grad, (0, 2, 3, 1)).reshape(-1,outc) / N
            col_dW = np.dot(col.T, grad)
            dW = np.transpose(col_dW, axes=(1, 0)).reshape(outc, inc, kh, kw)
            return dW
        depends_on.append(Dependency(kernel,BackwardConv2dKernel))

    if bias.requires_grad:
        def BackwardConv2dBias(grad:np.ndarray):
            grad = np.transpose(grad, (0, 2, 3, 1)).reshape(-1,outc)
            col_dB = np.sum(grad,axis=0,keepdims=True) / N
            return col_dB
        depends_on.append(Dependency(bias,BackwardConv2dBias))

    if x.requires_grad:
        def BackwardConv2dX(grad:np.ndarray):
            grad = grad.transpose(0, 2, 3, 1).reshape(-1,outc)
            dx = grad @ W.T
            d_in = _col2im(dx, X.shape, kh, kw, stride, out_h, out_w)
            return d_in
        depends_on.append(Dependency(x,BackwardConv2dX))

    return Tensor(out,kernel.requires_grad,depends_on)

def Linear(x,weights,bias):
    depends_on: List[Dependency] = []
    out = np.dot(x.data,weights.data)
    if bias:
        out += bias.data
    if x.requires_grad:
        def BackwardLinearX(grad:np.ndarray):
            nonlocal weights
            return grad @ weights.data.T
        depends_on.append(Dependency(x,BackwardLinearX))
    if weights.requires_grad:
        def BackwardLinearWeights(grad:np.ndarray):
            nonlocal x
            return x.data.T @ grad
        depends_on.append(Dependency(weights,BackwardLinearWeights))
    if bias.requires_grad:
        def BackwardLinearBias(grad:np.ndarray):
            return np.sum(grad, axis=0)
        depends_on.append(Dependency(bias,BackwardLinearBias))
    requires_grad = x.requires_grad or weights.requires_grad or bias.requires_grad
    return Tensor(out,requires_grad=requires_grad,depends_on=depends_on)

def ReLU(x):
    depends_on: List[Dependency] = []
    if x.requires_grad:
        def BackwardReLU(grad:np.ndarray):
            return np.where(x.data>0,grad,0)
        depends_on.append(Dependency(x,BackwardReLU))
    return Tensor(np.maximum(0,x.data),x.requires_grad,depends_on=depends_on)

def CrossEntropy(y_pred,y):
    eps = np.finfo(float).eps
    cross_entropy = -np.sum(y.data * np.log(y_pred.data + eps)) / y.shape[0]

    depends_on: List[Dependency] = []
    if y_pred.requires_grad:
        def CrossEntropyGradNative(grad:np.ndarray):
            return (y_pred.data - y.data)
        depends_on.append(Dependency(y_pred,CrossEntropyGradNative))
    return Tensor(cross_entropy,requires_grad=y_pred.requires_grad,depends_on=depends_on)

def BatchNorm2d(x,running_mean,running_var,beta,gamma,momentum,training,eps):
    if training:
        mu = np.mean(x.data, axis = (0,2,3), keepdims=True)
        var = np.var(x.data, axis = (0,2,3),keepdims=True) + eps
        std = np.sqrt(var)

        xhat = (x.data - mu) / std
        out = gamma.data * xhat + beta.data

        running_mean[:,:,:,] = momentum * running_mean + (1.0 - momentum) * mu
        running_var[:,:,:,] = momentum * running_var + (1.0 - momentum) * var
    else:
        mu = running_mean
        var = running_var
        norm_x = (x.data - mu) / np.sqrt(var + eps)
        out = gamma.data * norm_x + beta.data

    depends_on: List[Dependency] = []
    if x.requires_grad:
        def BackwardBatchNorm2dX(grad:np.ndarray):
            nonlocal x
            #部分代码引用自numpy-ml
            X = np.reshape(x.data, (-1, x.data.shape[1]))
            grad = np.reshape(grad, (-1, grad.shape[1]))
            n_ex, _ = X.shape
            X_mean, X_var = X.mean(axis=0), X.var(axis=0)

            N = (X - X_mean) / np.sqrt(X_var + eps)
            dN = grad * gamma.data.reshape(1, -1)
            dX = (n_ex * dN - dN.sum(axis=0) - N * (dN * N).sum(axis=0)) / (
                n_ex * np.sqrt(X_var + eps)
            )
            return dX.reshape(x.shape)

        def BackwardBatchNorm2dBeta(grad:np.ndarray):
            grad = np.reshape(grad, (-1, grad.shape[1]))
            return np.sum(grad, axis=0, keepdims=True).reshape(beta.shape)

        def BackwardBatchNorm2dGamma(grad:np.ndarray):
            nonlocal x
            grad = np.reshape(grad, (-1, grad.shape[1]))
            X = np.reshape(x.data, (-1, x.shape[1]))
            X_mean, X_var = X.mean(axis=0), X.var(axis=0)
            N = (X - X_mean) / np.sqrt(X_var + eps)
            return np.sum(grad*N, axis=0, keepdims=True).reshape(gamma.shape)

        depends_on.append(Dependency(x,BackwardBatchNorm2dX))
        depends_on.append(Dependency(gamma,BackwardBatchNorm2dGamma))
        depends_on.append(Dependency(beta,BackwardBatchNorm2dBeta))

    return Tensor(out,requires_grad=True,depends_on=depends_on)

def Softmax(x,dim,cross_entropy):
    ex = np.exp(x.data - np.max(x.data, axis=dim, keepdims=True))
    Y = ex / np.sum(ex,axis=dim,keepdims=True)

    depends_on: List[Dependency] = []
    if x.requires_grad:
        def BackwardSoftmax(grad:np.ndarray):
            if cross_entropy:
                return grad
            else:
                nonlocal x
                dX = []
                ex = lambda X:np.exp(X - np.max(X, axis=dim, keepdims=True))
                f = lambda X: ex(X)/ np.sum(X,axis=dim,keepdims=True)

                for dy, x in zip(grad, x.data):
                    dxi = []
                    for dyi, xi in zip(*np.atleast_2d(dy, x)):
                        yi = f(xi.reshape(1, -1)).reshape(-1, 1)
                        dyidxi = np.diagflat(yi) - yi @ yi.T  # jacobian wrt. input sample xi
                        dxi.append(dyi @ dyidxi)
                    dX.append(dxi)

                return np.array(dX).reshape(*x.data.shape)
        depends_on.append(Dependency(x,BackwardSoftmax))

    return Tensor(Y,x.requires_grad,depends_on=depends_on)

def Dropout(x,p,training):
    if not training:
        return x * (1-p) #使期望与训练时保持一致           
    if p == 0:
        return x
    else:
        mask = np.random.binomial(1, 1-p, size=x.shape)
        if x.requires_grad:
            def BackwardDropout(grad:np.ndarray) -> np.ndarray:
                return mask * grad
        return Tensor(x.data * mask,requires_grad=x.requires_grad,depends_on=[Dependency(x,BackwardDropout)])
