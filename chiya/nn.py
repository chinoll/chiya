from .autograd import Tensor
import numpy as np
import traceback
from typing import Iterator, Any
import inspect

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
