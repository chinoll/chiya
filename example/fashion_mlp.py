import numpy as np
import matplotlib.pyplot as plt
import tqdm
import utils
import sys
sys.path.append(utils.get_chiya_path())
from chiya import nn,Tensor,optim,utils
from chiya import dataset as td
from scipy.interpolate import interp1d

np.random.seed(42)
f = nn.Sequential(*[
    nn.flatten(),
    nn.Linear(200),
    nn.ReLU(),
    # nn.Linear(70),
    # nn.ReLU(),
    # nn.Linear(30),
    # nn.ReLU(),
    nn.Linear(10),
    # nn.Softmax()
])
def gradient_check(f,x,y,loss_func):
    xi = Tensor(x[0].reshape(1,1,28,28).astype(np.float32)/255)
    label = Tensor(utils.onehot(y[:1],10))
    F = f(xi)
    loss = loss_func(F,label)
    loss.backward()

    dreal = []
    for i in f.parameters():
        dreal.append(i.grad.data.reshape(-1))
    dreal = np.concatenate(dreal)

    # param = list(f.parameters())
    print("start gradient check")
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
    print("gradient check:",diff)
    f.zero_grad()
def train(f,x,y,lr=0.001,epochs=500,batch_size=2000):
    loss_list = []
    loss_func = nn.SoftmaxCrossEntropy()
    optimizer = optim.Adam(lr)
    for _ in tqdm.tqdm(range(epochs)):
        # xi = Tensor(x[0].reshape(1,1,28,28).astype(np.float32)/255)
        # label = Tensor(utils.onehot(y[:1],10))
        # F = f(xi)
        # loss = loss_func(F,label)
        # loss.backward()
        gradient_check(f,x,y,loss_func)
        # f.zero_grad()
        for xi,yi in utils.data_iter(x,y,batch_size):
            xi = xi.reshape(len(xi),-1).astype(np.float32)/255
            label = utils.onehot(yi,10)
            xi = Tensor(xi)
            label = Tensor(label)
            F = f(xi)
            loss = loss_func(F,label)
            loss.backward()
            optimizer.step(f)
            f.zero_grad()
            loss_list.append(loss.data)
    return loss_list

epochs = 30
(train_x,train_y),(test_x,test_y) = td.gen_mnist_dataset()
loss_list = train(f,train_x,train_y,lr=0.01,epochs=epochs)
test_x = Tensor(test_x,requires_grad=False)
print("acc",np.mean(np.argmax(f(test_x).data,axis=1)==test_y))

y = np.array(loss_list)
x = np.arange(len(y))
cubic_interploation_model=interp1d(x,y,kind="cubic")
x=np.linspace(0,epochs,epochs)
y = cubic_interploation_model(x)
plt.plot(x,y,color='red',label='loss')
plt.legend()
plt.show()

