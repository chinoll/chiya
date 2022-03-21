from .autograd import Tensor
import numpy as np
from typing import Iterator, Any
import inspect
from . import functional as F
from . import init

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

    def apply(self, func: callable, *args: Any, **kwds: Any):
        self.init_func = lambda x: func(x, *args, **kwds)
        for i in self.parameters():
            self.init_func(i)
            # i.zero_grad()
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

class Linear(Module):
    def __init__(self,out_features,in_features=None,bias=True):
        super(Linear,self).__init__()

        self.out_features = out_features
        if in_features == None:
            self.weight = None
            self.in_features = None
        else:
            self.in_features = in_features
            self.weight = Parameter(np.empty((in_features, out_features)),requires_grad=True)
        if bias:
            self.b = Parameter(np.zeros(out_features),requires_grad=True)
        else:
            self.b = None

    def apply(self, func: callable, *args: Any, **kwds: Any):
        if self.weight is not None:
            self.init_func(self.weight)
            # self.weight.zero_grad()
        else:
            self.init_func = lambda x: func(x, *args, **kwds)

    def __repr__(self):
        if self.in_features:
            return f"Linear(in_features={self.out_features},out_features={self.weight.shape[1]})"
        else:
            return f"Linear(out_features={self.out_features})"

    def forward(self, inputs):
        if self.weight is None:
            in_features = np.prod(inputs.shape[1:])
            self.in_features = in_features
            self.weight = Parameter(np.empty((in_features, self.out_features)),requires_grad=True)
            self.init_func(self.weight)
            print("Linear: init weight")
            print(self.weight.shape,self.b.shape,"linear")
        return F.Linear(inputs,self.weight,self.b)

class ReLU(Module):
    def __init__(self):
        super(ReLU,self).__init__()

    def __repr__(self) -> str:
        return f"ReLU()"

    def forward(self, inputs):
        return F.ReLU(inputs)

class CrossEntropy(Module):
    def __init__(self):
        super(CrossEntropy,self).__init__()

    def __repr__(self) -> str:
        return "CrossEntropy()"

    def forward(self,y_pred,y):
        return F.CrossEntropy(y_pred,y)

class Softmax(Module):
    def __init__(self, dim=-1,cross_entropy=False):
        super(Softmax,self).__init__()

        self.dim = dim
        self.n_in = None
        self.cross_entropy = cross_entropy

    def __repr__(self) -> str:
        return "Softmax()"

    def forward(self, X):
        return F.Softmax(X,self.dim,self.cross_entropy)

class Sequential(Module):
    def __init__(self, *layers):
        super(Sequential,self).__init__()

        self.layers = layers

    def __repr__(self) -> str:
        return 'Sequential(\n{}\n)'.format('\t'+', \n\t'.join(map(repr, self.layers)))

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

    def apply(self, func: callable, *args: Any, **kwds: Any):
        for layer in self.layers:
            layer.apply(func, *args, **kwds)

    def eval(self):
        for layer in self.layers:
            layer.eval()

class Conv2d(Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,bias=True):
        super(Conv2d,self).__init__()

        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)
        elif not isinstance(kernel_size,(tuple,list)):
            raise TypeError("kernel_size must be int or tuple/list")

        if isinstance(stride,int):
            stride = (stride,stride)
        elif not isinstance(stride,(tuple,list)):
            raise TypeError("stride must be int or tuple/list")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = Parameter(np.empty((out_channels,in_channels,kernel_size[0],kernel_size[1])).astype('float32'),requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.bias = Parameter(np.zeros((1,out_channels)).astype('float32'),requires_grad=bias)

    def __repr__(self):
        return f"Conv2d(in_channels={self.in_channels},out_channels={self.out_channels},kernel_size={self.kernel_size},stride={self.stride},padding={self.padding})"

    def apply(self, func: callable, *args: Any, **kwds: Any):
        func(self.kernel, *args, **kwds)

    def forward(self, x):
        if self.padding:
            x = x.pad(((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))
        #print kernel and bias shape
        # print(self.kernel.shape,self.bias.shape,"conv2d")
        return F.Conv2d(x,self.kernel,self.bias,self.stride)

class flatten(Module):
    def __repr__(self):
        return f"flatten()"

    def forward(self,x):
        return x.reshape(x.shape[0],-1)

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0, padding_mode='zero'):
        super(MaxPool2d,self).__init__()

        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)
        elif not isinstance(kernel_size,(tuple,list)):
            raise TypeError('kernel_size must be int or tuple/list')

        if isinstance(stride,int):
            stride = (stride,stride)
        elif not isinstance(stride,(tuple,list)):
            raise TypeError('stride must be int or tuple/list')

        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size

    def __repr__(self) -> str:
        return f"MaxPool2d(kernel_size={self.kernel_size},stride={self.stride},padding={self.padding})"

    def forward(self, x):
        if self.padding_mode == 'zero':
            x = x.pad(((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))

        return F.MaxPool2d(x,self.kernel_size,self.stride)

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0, padding_mode='zero'):
        super(AvgPool2d,self).__init__()

        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)
        elif not isinstance(kernel_size,(tuple,list)):
            raise TypeError('kernel_size must be int or tuple/list')

        if isinstance(stride,int):
            stride = (stride,stride)
        elif not isinstance(stride,(tuple,list)):
            raise TypeError('stride must be int or tuple/list')

        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode

    def __repr__(self) -> str:
        return f"AvgPool2d(kernel_size={self.kernel_size},stride={self.stride},padding={self.padding})"

    def forward(self, x):
        if self.padding_mode == 'zero':
            x = x.pad(((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))

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
        self.num_features = num_features

    def __repr__(self) -> str:
        return f"BatchNorm2d(num_features={self.num_features},eps={self.eps},momentum={self.momentum})"

    def apply(self, func: callable, *args: Any, **kwds: Any):
        #不对gamma和beta初始化
        pass

    def forward(self,x):
        return F.BatchNorm2d(x,self.running_mean,self.running_var,self.beta,self.gamma,self.momentum,self.training,self.eps)

class Dropout(Module):
    def __init__(self,p=0.5):
        super(Dropout,self).__init__()
        self.p = p

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"

    def forward(self,x:Tensor) -> Tensor:
        return F.Dropout(x,self.p,self.training)