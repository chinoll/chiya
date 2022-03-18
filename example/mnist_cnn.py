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
    nn.Conv2d(1,16,3,1,False),
    nn.ReLU(),
    nn.BatchNorm2d(16),
    nn.AvgPool2d(2,2),
    nn.flatten(),
    nn.linear(16*13*13,1024),
    nn.ReLU(),
    nn.linear(1024,10),
    nn.softmax()
])

def train(f,x,y,lr=0.001,epochs=500,batch_size=1000):
    loss_list = []
    loss_func = nn.cross_entropy()
    optimizer = optim.Adam(lr)
    for _ in tqdm.tqdm(range(epochs)):
        for xi,yi in utils.data_iter(x,y,batch_size):
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

epochs = 2
(train_x,train_y),(test_x,test_y) = td.gen_mnist_dataset()
loss_list = train(f,train_x,train_y,lr=0.01,epochs=epochs)
test_x = Tensor(test_x.reshape(-1,1,28,28).astype(np.float32)/255,requires_grad=False)
f.eval()
print("acc",np.mean(np.argmax(f(test_x).data,axis=1)==test_y))

y = np.array(loss_list)
x = np.arange(len(y))
plt.plot(x,y,color='red',label='loss')
plt.legend()
plt.show()