import numpy as np
def calculate_gain(nonlinearity,param=None):
    #把字符串转换为小写
    nonlinearity = nonlinearity.lower()
    if nonlinearity == 'relu':
        return 1.4142135623730951 # sqrt(2)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            raise ValueError("Param for LeakyReLU must not be None")
        if param == 0.01:
            return 1.4141428569978354 # sqrt(2/(1+0.01**2))
        elif param == 0.02:
            return 1.4139308044851597 # sqrt(2/(1+0.02**2))
        elif param == 0.03:
            return 1.4135775955154748 # sqrt(2/(1+0.03**2))
        elif param == 0.04:
            return 1.4130835473605536 # sqrt(2/(1+0.04**2))
        elif param == 0.05:
            return 1.4124491030928974 # sqrt(2/(1+0.05**2))
        else:
            return np.sqrt(2/(1+param**2))
    elif nonlinearity.startswith('conv'):
        return 1.0
    elif nonlinearity == 'sigmoid':
        return 1.0
    elif nonlinearity == 'tanh':
        return 1.666 #5/3
    elif nonlinearity == 'linear':
        return 1.0
    elif nonlinearity == 'elu':
        return 1.0
    else:
        raise ValueError("Unsupported nonlinearity: {}".format(nonlinearity))

def uniform_(tensor, a=0, b=1):
    tensor.data = np.random.uniform(a, b, tensor.shape)
    return tensor

def normal_(tensor, mean=0, std=1):
    tensor.data = np.random.normal(mean, std, tensor.shape) * 0.001
    return tensor

def constant_(tensor, val=0):
    tensor.data = np.ones(tensor.shape)*val
    return tensor

def ones_(tensor):
    tensor.data = np.ones(tensor.shape)
    return tensor

def zeros_(tensor):
    tensor.data = np.zeros(tensor.shape)
    return tensor

def eye_(tensor):
    tensor.data = np.eye(tensor.shape[0])
    return tensor

def _calculate_fan_in_and_fan_out(tensor):
    if tensor.data.ndim < 2:
        raise ValueError("fan in and fan out can not be computed for tensor with less than 2 dimensions")

    fan_in = tensor.shape[0] if len(tensor.shape) == 2 else np.prod(tensor.shape[1:])
    fan_out = tensor.shape[1] if len(tensor.shape) == 2 else tensor.shape[0]
    return fan_in, fan_out

def xavier_uniform_(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    a = gain * np.sqrt(6.0 / (fan_in + fan_out))
    print(a)
    tensor.data = np.random.uniform(-a, a, tensor.shape)
    return tensor

def xavier_normal_(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data = np.random.normal(0, std, tensor.shape)
    return tensor

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    gain = calculate_gain(nonlinearity, a)
    if mode == 'fan_in':
        bound = gain * np.sqrt(3 / fan_in)
    elif mode == 'fan_out':
        bound = gain * np.sqrt(3 / fan_out)

    tensor.data = np.random.uniform(-bound, bound, tensor.data.shape)
    print(tensor.data)
    return tensor

def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    gain = calculate_gain(nonlinearity, a)
    if mode == 'fan_in':
        std = gain / np.sqrt(fan_in)
    elif mode == 'fan_out':
        std = gain / np.sqrt(fan_out)

    tensor.data = np.random.normal(0, std, tensor.shape)
    return tensor
