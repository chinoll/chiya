from .nn import Module
import numpy as np
from queue import Queue
from .autograd import Tensor
class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, module: Module) -> None:
        for parameter in module.parameters():
            parameter -= parameter.grad * self.lr

class Adam:
    def __init__(self, lr:float = 0.001,beta:tuple = (0.9,0.99),eps:float =1e-8) -> None:
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.ms = Queue()
        self.vs = Queue()
        self.t = 0
    def step(self, module: Module) -> None:
        if len(self.ms.queue) == 0:
            for parameter in module.parameters():
                self.ms.put(Tensor(np.zeros(parameter.shape)))
                self.vs.put(Tensor(np.zeros(parameter.shape)))
        self.t += 1
        for ms,vs,parameter in zip(self.ms.queue,self.vs.queue,module.parameters()):
            ms = self.beta[0]*ms + (1-self.beta[0])*parameter.grad
            vs =  (1-self.beta[1])*parameter.grad*parameter.grad + self.beta[1]*vs
            m:Tensor = ms/(1-self.beta[0]**self.t)
            v:Tensor = vs/(1-self.beta[1]**self.t)
            parameter -= self.lr*m/(np.sqrt(v)+self.eps)