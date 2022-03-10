import numpy as np
import matplotlib.pyplot as plt
import tqdm
import utils
import sys
sys.path.append(utils.get_chiya_path())
from chiya import nn,Tensor,optim,utils
from chiya import dataset as td
np.random.seed(42)
f = nn.Sequential(*[
    nn.Conv2d(1,8,3,1,bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    # nn.Conv2d(8,16,2,2,bias=False),
    # nn.ReLU(),
    nn.flatten(),
    nn.linear(8*13*13,10),
    nn.softmax()
])

def data_iter(x,y,batch_size=1):
    m = len(x)
    idx = np.arange(m)
    for i in range(0,m,batch_size):
        j = min(i+batch_size,m)
        yield x[idx[i:j]],y[idx[i:j]]

def train(f,x,y,lr=0.001,epochs=500,batch_size=1000):
    loss_list = []
    loss_func = nn.cross_entropy()
    optimizer = optim.Adam(lr)
    for _ in tqdm.tqdm(range(epochs)):
        for xi,yi in data_iter(x,y,batch_size):
            xi = xi.reshape(len(xi),-1,28,28).astype(np.float32)/255
            label = utils.onehot(yi,10)
            xi = Tensor(xi)
            label = Tensor(label)
            F = f(xi)
            loss = loss_func(F,label)
            print(loss)

            loss.backward()
            optimizer.step(f)
            f.zero_grad()
            loss_list.append(loss.data)
    return loss_list

(train_x,train_y),(test_x,test_y) = td.gen_mnist_dataset()
loss_list = train(f,train_x,train_y,lr=0.01,epochs=10)
test_x = Tensor(test_x.reshape(-1,1,28,28).astype(np.float32)/255,requires_grad=False)
print("acc",np.mean(np.argmax(f(test_x).data,axis=1)==test_y))

plt.plot([i for i in range(len(loss_list))],[i for i in loss_list],color='red',label='loss')
plt.legend()

plt.show()

