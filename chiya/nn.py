from .autograd import Tensor
import numpy as np
from typing import Iterator, Any
import inspect
from . import functional as F

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
        if self.weight is None:
            in_featues = np.prod(inputs.shape[1:])
            self.weight = Parameter(normal((in_featues, self.out_features)),requires_grad=True)
        return F.Linear(inputs,self.weight,self.b)

class ReLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        return F.ReLU(inputs)

class CrossEntropy(Module):
    def __init__(self):
        super().__init__()

    def forward(self,y_pred,y):
        return F.CrossEntropy(y_pred,y)

class Softmax(Module):
    def __init__(self, dim=-1,cross_entropy=False):
        super().__init__()

        self.dim = dim
        self.n_in = None
        self.cross_entropy = cross_entropy

    def forward(self, X):
        return F.Softmax(X,self.dim,self.cross_entropy)

class Sequential(Module):
    def __init__(self, *layers):
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

def pad_image(img:np.ndarray,m:int,n:int)->np.ndarray:
    return np.pad(img,((0,0),(0,0),(m,m),(n,n)),mode='constant',constant_values=0)

class Conv2d(Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True):
        super().__init__()
        self.kernel = Parameter(normal((out_channels,in_channels,kernel_size,kernel_size)).astype('float32'),requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.bias = Parameter(np.zeros((1,out_channels)).astype('float32'),requires_grad=bias)
    def forward(self, x):
        x = x.pad(((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        return F.Conv2d(x,self.kernel,self.bias,self.stride)

class flatten(Module):
    def forward(self,x):
        return x.reshape(x.shape[0],-1)

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0, padding_mode='zero'):
        super().__init__()
        self.kernel_size = (kernel_size,kernel_size) if type(kernel_size) == int else kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, x):
        return F.MaxPool2d(x,self.kernel_size,self.stride)

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0, padding_mode='zero'):
        super().__init__()
        self.kernel_size = (kernel_size,kernel_size) if type(kernel_size) == int else kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

    def forward(self, x):
        return F.AvgPool2d(x,self.kernel_size,self.stride)

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
        return F.BatchNorm2d(x,self.running_mean,self.running_var,self.beta,self.gamma,self.momentum,self.training,self.eps)

def normal(shape):
    return np.random.normal(size=shape)*0.001