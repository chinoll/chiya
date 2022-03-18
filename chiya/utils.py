import numpy as np
from .autograd import Tensor

def onehot(label,class_num):
    one_hot = np.zeros((label.shape[0],class_num))
    one_hot[np.arange(label.shape[0]),label] = 1
    return one_hot

def data_iter(x,y,batch_size=1):
    m = len(x)
    idx = np.arange(m)
    for i in range(0,m,batch_size):
        j = min(i+batch_size,m)
        yield x[idx[i:j]],y[idx[i:j]]

def gradient_check(f,x,y,loss_func):
    xi = Tensor(x[0].reshape(1,1,28,28).astype(np.float32)/255)
    label = Tensor(onehot(y[:1],10))
    F = f(xi)
    loss = loss_func(F,label)
    loss.backward()

    dreal = []
    for i in f.parameters():
        dreal.append(i.grad.data.reshape(-1))
    dreal = np.concatenate(dreal)

    param = []
    size = 0
    for i in f.parameters():
        param.append(i.data.copy().reshape(-1))
        size += i.data.size
    dapprox = np.zeros(size)
    count = 0
    for ip,pp in zip(f.parameters(),param):
        for j in range(ip.data.size):
            a = pp.copy()
            a[j] += 1e-7
            ip.data.data = a.reshape(ip.shape).data
            loss1 = loss_func(f(xi),label).data

            a[j] -= 2*1e-7
            ip.data.data = a.reshape(ip.shape).data
            loss2 = loss_func(f(xi),label).data
            dapprox[count] = (loss1-loss2)/1e-7/2
            count += 1
    diff = np.linalg.norm(dreal-dapprox)/(np.linalg.norm(dreal)+np.linalg.norm(dapprox))
    f.zero_grad()
    return diff