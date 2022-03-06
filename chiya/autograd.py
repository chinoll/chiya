from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np

#Part of the code comes from https://github.com/joelgrus/autograd
class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]

def _ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

Tensorable = Union['Tensor', float, np.ndarray]

def _ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(self,
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        self._data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self._data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # Setting the data manually means we invalidate the gradient.
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """gets called if I do t + other"""
        return _add(self, _ensure_tensor(other))

    # __raddr__ = __add__
    def __radd__(self, other) -> 'Tensor':
        """gets called if I do other + t"""
        print("radd")
        tensor = _add(_ensure_tensor(other), self)
        print(tensor)
        return tensor

    def __iadd__(self, other) -> 'Tensor':
        """when we do t += other"""
        self.data = self.data + _ensure_tensor(other).data
        return self

    def __isub__(self, other) -> 'Tensor':
        """when we do t -= other"""
        self.data = self.data - _ensure_tensor(other).data
        return self

    def __imul__(self, other) -> 'Tensor':
        """when we do t *= other"""
        self.data = self.data * _ensure_tensor(other).data
        return self

    def __mul__(self, other) -> 'Tensor':
        return _mul(self, _ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return _mul(_ensure_tensor(other), self)

    def __matmul__(self, other) -> 'Tensor':
        return _matmul(self, other)

    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __sub__(self, other) -> 'Tensor':
        return _sub(self, _ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return _sub(_ensure_tensor(other), self)
    def __truediv__(self, other) -> 'Tensor':
        return _div(self, _ensure_tensor(other))
    def __rtrueiv__(self, other) -> 'Tensor':
        return _div(_ensure_tensor(other), self)
    def __getitem__(self, idxs) -> 'Tensor':
        return _slice(self, idxs)
    
    def exp(self):
        return _exp(self)
    def log(self):
        return _log(self)
    def max(self,axis=None,keepdims=False):
        return _max(self,axis,keepdims)
    def maximum(self,other) -> 'Tensor':
        return _maximum(self,_ensure_tensor(other))
    def sqrt(self):
        return _sqrt(self)
    def transpose(self,*axes):
        return _transpose(self,*axes)
    def copy(self) -> 'Tensor':
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)
    def reshape(self,*shape):
        return _reshape(self,*shape)
    def flatten(self) -> 'Tensor':
        return self.reshape(self.shape[0],-1)

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")
        self.grad.data = self.grad.data + grad.data  # type: ignore

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self,axis=None,keepdims=False) -> 'Tensor':
        return tensor_sum(self,axis=axis,keepdims=keepdims)


def tensor_sum(t: Tensor,axis=None,keepdims=False) -> Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """
    data = t.data.sum(axis=axis,keepdims=keepdims)
    requires_grad = t.requires_grad
    depends_on: List[Dependency] = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each input element
            contributes that much
            """
            return grad * np.ones_like(t.data)

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data,requires_grad,depends_on)

def _reshape(t: Tensor,*shape):
    raw_shape = t.data.shape
    data = t.data.reshape(*shape)
    requires_grad = t.requires_grad
    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.reshape(*raw_shape)
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data,requires_grad,depends_on)

def _transpose(t: Tensor,*axes) -> Tensor:
    data = t.data.transpose(*axes)
    requires_grad = t.requires_grad
    depends_on: List[Dependency] = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad.transpose(*axes)
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data,requires_grad,depends_on)

def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,requires_grad,depends_on)

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data

            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data

            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,requires_grad,depends_on)

def _exp(t1: Tensor) -> Tensor:
    data = np.exp(t1.data)
    requires_grad = t1.requires_grad
    depends_on: List[Dependency] = []
    if t1.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.exp(data)
        depends_on.append(Dependency(t1, grad_fn))
    return Tensor(data,requires_grad,depends_on)

def _div(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data / t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []
    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad / t2.data
        depends_on.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return -grad * data / (t2.data ** 2)
        depends_on.append(Dependency(t2, grad_fn2))
    return Tensor(data,requires_grad,depends_on)

def _log(t1: Tensor) -> Tensor:
    data = np.log(t1.data)
    requires_grad = t1.requires_grad
    depends_on: List[Dependency] = []
    if t1.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad / t1.data
        depends_on.append(Dependency(t1, grad_fn))
    return Tensor(data,requires_grad,depends_on)


def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2

def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,requires_grad,depends_on)

def _slice(t: Tensor, idxs) -> Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idxs] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def repeat_to_match_shape(g, shape, dtype, axis, keepdims):
    """Returns the array g repeated along axis to fit vector space vs.
       Also returns the number of repetitions of the array."""
    if shape == ():
      return g, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = np.array(shape)
    new_shape[axis] = 1
    num_reps = np.prod(np.array(shape)[axis])
    # Can't use broadcast_to because of numpy bug: https://github.com/numpy/numpy/issues/9165
    # return anp.broadcast_to(anp.reshape(g, new_shape), shape), num_reps
    return np.reshape(g, new_shape) + np.zeros(shape, dtype=dtype), num_reps

def grad_chooser(ans, x, axis=None, keepdims=None):
    shape, dtype = np.shape(x), np.result_type(x)
    def vjp(g):
        """Builds gradient of functions that choose a single item, such as min or max."""
        g_repeated, _ = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        argmax_locations = x == repeat_to_match_shape(ans, shape, dtype, axis, keepdims)[0]
        return g_repeated * argmax_locations / np.sum(argmax_locations, axis=axis, keepdims=True)
    return vjp

def _max(t: Tensor, axis=None, keepdims=False) -> Tensor:
    data = np.max(t.data, axis=axis, keepdims=keepdims)
    requires_grad = t.requires_grad
    depends_on: List[Dependency] = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # This implementation may be wrong, but it is useful and can speed up convergence.
            max_grad = np.zeros_like(t.data)
            bool_array = t.data == np.max(t.data,keepdims=True,axis=axis)
            max_grad[bool_array] = 1 / bool_array.sum()
            return max_grad
            # return grad_chooser(data, t.data, axis, keepdims)(grad)
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data, requires_grad, depends_on)

def balanced_eq(x, y, z):
    return (x == z) / (1.0 + (x == y))

def _maximum(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.maximum(t1.data, t2.data)
    requires_grad = t1.requires_grad or (isinstance(t2, Tensor) and t2.requires_grad)
    depends_on: List[Dependency] = []
    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return balanced_eq(data,t1.data, t2.data) * grad
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return balanced_eq(data,t2.data, t1.data) * grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)

def _sqrt(t: Tensor) -> Tensor:
    data = np.sqrt(t.data)
    requires_grad = t.requires_grad
    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return (grad*0.5)*(data**-0.5)
        depends_on.append(Dependency(t, grad_fn))
    return Tensor(data, requires_grad, depends_on)