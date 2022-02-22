import numpy

import numpy as np

#算子
class OP:
    def forward(self, tensors):
        pass

    def backward(self, tensors, grad):
        pass


class Add(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data + tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        return [grad, grad]

class MulOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data * tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        return [grad * tensors[1].data, grad * tensors[0].data]

class DivOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data / tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        return [grad / tensors[1].data,-grad * tensors[0].data / tensors[1].data**2]

class SubOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data - tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        return [grad,-grad]

class AddConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[1],(int,float)):
            return Tensor(tensors[0].data + tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] + tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        return [1]

class MulConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[1],(int,float)):
            return Tensor(tensors[0].data * tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] * tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        if isinstance(tensors[0],(int,float)):
            return [tensors[0]]
        else:
            return [tensors[1]]

class DivConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[1],(int,float)):
            return Tensor(tensors[0].data / tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] / tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        if isinstance(tensors[0],(int,float)):
            return [1/tensors[0]]
        else:
            return [1/tensors[1]]

class SubConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[1],(int,float)):
            return Tensor(tensors[0].data - tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] - tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        return [1]

class NegOp(OP):
    def forward(self, tensors):
        return Tensor(-tensors[0].data,tensors,self)

    def backward(self, tensors, grad):
        return [-1]

class PowOp(OP):
    def forward(self, tensors):
        self.result = tensors[0].data ** tensors[1].data
        return Tensor(self.result,tensors,self)

    def backward(self, tensors, grad):
        print(tensors[1].data)
        return [ tensors[1].data*tensors[0].data**(tensors[1].data-1),
                 self.result*np.log(tensors[0].data)]

class PowConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[1],(int,float)):
            self.result = tensors[0].data ** tensors[1]
            self.pow = False
        else:
            self.result = tensors[0] ** tensors[1].data
            self.pow = True
        return Tensor(self.result, tensors, self)

    def backward(self, tensors, grad):
        if not self.pow:
            return [tensors[1]*tensors[0].data**(tensors[1]-1)]
        else:
            return [self.result*np.log(tensors[0])]

class ModConst(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data % tensors[1], tensors, self)

    def backward(self, tensors, grad):
        return [1]

class EqOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data == tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'==' operator is not autograd")

class NeOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data != tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'!=' operator is not autograd")

class LtOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data < tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'<' operator is not autograd")

class LeOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data <= tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'<=' operator is not autograd")

class GtOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data > tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'>' operator is not autograd")

class GeOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data >= tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'>=' operator is not autograd")

class AndOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data & tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'&' operator is not autograd")

class OrOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data | tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'|' operator is not autograd")

class NotOp(OP):
    def forward(self, tensors):
        return Tensor(~tensors[0].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("'~' operator is not autograd")

class CeilOp(OP):
    def forward(self, tensors):
        return Tensor(np.ceil(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [0.]

class FloorOp(OP):
    def forward(self, tensors):
        return Tensor(np.floor(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [0.]

class LshiftOp(OP):
    def forward(self, tensors):
        return Tensor(np.left_shift(tensors[0].data, tensors[1].data), tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("lshiftOp.backward() is not autograd")

class RshiftOp(OP):
    def forward(self, tensors):
        return Tensor(np.right_shift(tensors[0].data, tensors[1].data), tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("rshiftOp.backward() is not autograd")

class InvertOp(OP):
    def forward(self, tensors):
        return Tensor(~tensors[0].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("InvertOp.backward() is not autograd")

class RoundOp(OP):
    def forward(self, tensors):
        return Tensor(np.round(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [0.]

class TruncOp(OP):
    def forward(self, tensors):
        return Tensor(np.round(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [0.]

class FloorDivOp(OP):
    def forward(self, tensors):
        return Tensor(np.floor_divide(tensors[0].data, tensors[1].data), tensors, self)

    def backward(self, tensors, grad):
        return [0.,0.]

class XorOp(OP):
    def forward(self, tensors):
        return Tensor(tensors[0].data ^ tensors[1].data, tensors, self)

    def backward(self, tensors, grad):
        raise RuntimeError("XorOp.backward() is not autograd")

class EqConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data == tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] == tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("EqConstantOp.backward() is not autograd")

class NeConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data != tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] != tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("NeqConstantOp.backward() is not autograd")

class LtConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data < tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] < tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("LtConstantOp.backward() is not autograd")

class GtConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data > tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] > tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("GtConstantOp.backward() is not autograd")

class LeConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data <= tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] <= tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("LeConstantOp.backward() is not autograd")

class GeConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data >= tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] >= tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("GeConstantOp.backward() is not autograd")

class AndConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data & tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] & tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("AndConstantOp.backward() is not autograd")

class OrConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data | tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] | tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("OrConstantOp.backward() is not autograd")

class XorConst(OP):
    def forward(self, tensors):
        if isinstance(tensors[0],Tensor):
            return Tensor(tensors[0].data ^ tensors[1], tensors, self)
        else:
            return Tensor(tensors[0] ^ tensors[1].data, tensors, self)
    def backward(self, tensors, grad):
        raise RuntimeError("XorConstantOp.backward() is not autograd")

class AbsOp(OP):
    def forward(self, tensors):
        return Tensor(np.abs(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [-1.]

class FloorDivConst(OP):
    def forward(self, tensors):
        return Tensor(np.floor_divide(tensors[0].data, tensors[1]), tensors, self)

    def backward(self, tensors, grad):
        return [0.]

class CosOp(OP):
    def forward(self, tensors):
        return Tensor(np.cos(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [-np.sin(tensors[0].data)]

class SinOp(OP):
    def forward(self, tensors):
        return Tensor(np.sin(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [np.cos(tensors[0].data)]

class TanOp(OP):
    def forward(self, tensors):
        return Tensor(np.tan(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./(np.cos(tensors[0].data)**2)]

class ArcCosOp(OP):
    def forward(self, tensors):
        return Tensor(np.arccos(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [-1./np.sqrt(1-tensors[0].data**2)]

class ArcSinOp(OP):
    def forward(self, tensors):
        return Tensor(np.arcsin(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./np.sqrt(1-tensors[0].data**2)]

class ArcTanOp(OP):
    def forward(self, tensors):
        return Tensor(np.arctan(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./(1+tensors[0].data**2)]

class SinhOp(OP):
    def forward(self, tensors):
        return Tensor(np.sinh(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [np.cosh(tensors[0].data)]

class CoshOp(OP):
    def forward(self, tensors):
        return Tensor(np.cosh(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [-np.sinh(tensors[0].data)]

class TanhOp(OP):
    def forward(self, tensors):
        return Tensor(np.tanh(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./(np.cosh(tensors[0].data)**2)]

class ArcsinhOp(OP):
    def forward(self, tensors):
        return Tensor(np.arcsinh(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./np.sqrt(tensors[0].data**2+1)]

class ArccoshOp(OP):
    def forward(self, tensors):
        return Tensor(np.arccosh(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./np.sqrt(tensors[0].data**2-1)]

class ArctanhOp(OP):
    def forward(self, tensors):
        return Tensor(np.arctan(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./(1-tensors[0].data**2)]

class SqrtOp(OP):
    def forward(self, tensors):
        return Tensor(np.sqrt(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [0.5/np.sqrt(tensors[0].data)]

class ExpOp(OP):
    def forward(self, tensors):
        return Tensor(np.exp(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [np.exp(tensors[0].data)]

class LogOp(OP):
    def forward(self, tensors):
        return Tensor(np.log(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./tensors[0].data]

class Log10Op(OP):
    def forward(self, tensors):
        return Tensor(np.log10(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./(tensors[0].data*np.log(10))]
class Log2Op(OP):
    def forward(self, tensors):
        return Tensor(np.log2(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./(tensors[0].data*np.log(2))]
class Log1pOp(OP):
    def forward(self, tensors):
        return Tensor(np.log1p(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [1./(1+tensors[0].data)]
class ZerosLikeOp(OP):
    def forward(self, tensors):
        return Tensor(np.zeros_like(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [0.]
class OnesLikeOp(OP):
    def forward(self, tensors):
        return Tensor(np.ones_like(tensors[0].data), tensors, self)

    def backward(self, tensors, grad):
        return [0.]
class MaxOp(OP):
    def forward(self, tensors):
        self._max = np.max(tensors[0].data)
        return Tensor(self._max, tensors, self)

    def backward(self, tensors, grad):
        grad = np.zeros_like(tensors[0].data)
        bool_array = tensors[0].data == self._max
        grad[bool_array] = 1/bool_array.sum()
        return [grad]
class MinOp(OP):
    def forward(self, tensors):
        self._min = np.min(tensors[0].data)
        return Tensor(self._min, tensors, self)

    def backward(self, tensors, grad):
        grad = np.zeros_like(tensors[0].data)
        bool_array = tensors[0].data == self._min
        grad[bool_array] = 1/bool_array.sum()
        return [grad]

class Tensor:
    def __init__(self, data, tensors=None, op=None, grad=None):
        self.data = data
        self.tensors = tensors
        self.op = op
        if grad:
            self.grad = grad
        else:
            self.grad = numpy.zeros(self.data.shape) if isinstance(self.data, numpy.ndarray) else 0
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    def __repr__(self) -> str:
        return 'tenor('+str(self.data)+')'
    def __add__(self, other):
        return Add().forward([self, other]) if isinstance(other, Tensor) else AddConst().forward([self, other])
    def __radd__(self, other):
        return Add().forward([other, self]) if isinstance(other, Tensor) else AddConst().forward([other, self])
    def __mul__(self,other):
        return MulOp().forward([self,other]) if isinstance(other, Tensor) else MulConst().forward([self, other])
    def __rmul__(self,other):
        return MulOp().forward([other,self]) if isinstance(other, Tensor) else MulConst().forward([other, self])
    def __truediv__(self,other):
        return DivOp().forward([self,other]) if isinstance(other, Tensor) else DivConst().forward([self, other])
    def __rtruediv__(self,other):
        return DivOp().forward([other,self]) if isinstance(other, Tensor) else DivConst().forward([other, self])
    def __sub__(self,other):
        return SubOp().forward([self,other]) if isinstance(other, Tensor) else SubConst().forward([self, other])
    def __rsub__(self,other):
        return SubOp().forward([other,self]) if isinstance(other, Tensor) else SubConst().forward([other, self])
    def __neg__(self):
        return NegOp().forward([self]) 
    def __pow__(self,other):
        return PowOp().forward([self,other]) if isinstance(other, Tensor) else PowConst().forward([self,other])
    def __rpow__(self,other):
        return PowOp().forward([other,self]) if isinstance(other, Tensor) else PowConst().forward([other,self])
    def __mod__(self,other):
        if isinstance(other, Tensor):
            raise RuntimeError("Tensor % Tensor is not supported, please use Tensor % number")
        return ModConst().forward([self,other])
    def __eq__(self,other):
        return EqOp().forward([self,other])
    def __ne__(self, other):
        return NeOp().forward([self,other]) if isinstance(other, Tensor) else NeConst().forward([self,other])
    def __lt__(self, other):
        return LtOp().forward([self, other]) if isinstance(other, Tensor) else LtConst().forward([self, other])
    def __le__(self, other):
        return LeOp().forward([self, other]) if isinstance(other, Tensor) else LtConst().forward([self, other])
    def __gt__(self, other):
        return GtOp().forward([self, other]) if isinstance(other, Tensor) else GtConst().forward([self, other])
    def __ge__(self, other):
        return GeOp().forward([self, other]) if isinstance(other, Tensor) else GeConst().forward([self, other])
    def __and__(self , other):
        return AndOp().forward([self,other]) if isinstance(other, Tensor) else AndConst().forward([self,other])
    def __rand__(self, other):
        return AndOp().forward([other,self]) if isinstance(other, Tensor) else AndConst().forward([other,self])
    def __or__(self, other):
        return OrOp().forward([self,other]) if isinstance(other, Tensor) else OrConst().forward([self,other])
    def __ror__(self, other):
        return OrOp().forward([other,self]) if isinstance(other, Tensor) else OrConst().forward([other,self])
    def __xor__(self, other):
        return XorOp().forward([self, other]) if isinstance(other, Tensor) else XorConst().forward([self, other])
    def __rxor__(self, other):
        return XorOp().forward([other, self]) if isinstance(other, Tensor) else XorConst().forward([other, self])
    def __floor__(self):
        return FloorOp().forward([self])
    def __ceil__(self):
        return CeilOp().forward([self])
    def __abs__(self):
        return AbsOp().forward([self])
    def __trunc__(self):
        return TruncOp().forward([self])
    def __round__(self):
        return RoundOp().forward([self])
    def __floordiv__(self, other):
        return FloorDivOp().forward([self, other]) if isinstance(other, Tensor) else FloorDivConst().forward([self, other])
    def cos(self):
        return CosOp().forward([self])
    def sin(self):
        return SinOp().forward([self])
    def tan(self):
        return TanOp().forward([self])
    def arccos(self):
        return ArcCosOp().forward([self])
    def arcsin(self):
        return ArcSinOp().forward([self])
    def arctan(self):
        return ArcTanOp().forward([self])
    def cosh(self):
        return CoshOp().forward([self])
    def sinh(self):
        return SinhOp().forward([self])
    def tanh(self):
        return TanhOp().forward([self])
    def arccosh(self):
        return ArccoshOp().forward([self])
    def arcsinh(self):
        return ArcsinhOp().forward([self])
    def arctanh(self):
        return ArctanhOp().forward([self])
    def exp(self):
        return ExpOp().forward([self])
    def log(self):
        return LogOp().forward([self])
    def log10(self):
        return Log10Op().forward([self])
    def log2(self):
        return Log2Op().forward([self])
    def log1p(self):
        return Log1pOp().forward([self])
    def sqrt(self):
        return SqrtOp().forward([self])
    def zeros_like(self):
        return ZerosLikeOp().forward([self])
    def ones_like(self):
        return OnesLikeOp().forward([self])
    def max(self):
        return MaxOp().forward([self])
    def min(self):
        return MinOp().forward([self])
    def backward(self, grad=None):
        if grad is None:
            self.grad = grad = numpy.ones(self.data.shape) if isinstance(self.data, numpy.ndarray) else 1
        if self.op:
            grad = self.op.backward(self.tensors, grad)
        if self.tensors:
            #清空常量
            self.tensors = [tensor for tensor in self.tensors if isinstance(tensor,Tensor)]

            for i in range(len(grad)):
                tensor = self.tensors[i]
                tensor.grad += grad[i]
                tensor.backward(grad[i])

    def zero_gard(self):
        self.grad = numpy.zeros(self.data.shape) if isinstance(self.data, numpy.ndarray) else 0