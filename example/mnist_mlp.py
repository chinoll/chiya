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
    nn.linear(784,200),
    nn.ReLU(),
    nn.linear(200,10),
    nn.softmax()
])

def train(f,x,y,lr=0.001,epochs=500,batch_size=1000):
    loss_list = []
    loss_func = nn.cross_entropy()
    optimizer = optim.Adam(lr)
    for _ in tqdm.tqdm(range(epochs)):
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

epochs = 100
(train_x,train_y),(test_x,test_y) = td.gen_mnist_dataset()
loss_list = train(f,train_x,train_y,lr=0.001,epochs=epochs)
test_x = Tensor(test_x,requires_grad=False)
print("acc",np.mean(np.argmax(f(test_x).data,axis=1)==test_y))

x = np.arange(len(loss_list))
y = np.array(loss_list)
cubic_interploation_model=interp1d(x,y,kind="cubic")
x = np.linspace(0,epochs,epochs)
y = cubic_interploation_model(x)
plt.plot(x,y,color='red',label='loss')
plt.legend()
plt.show()

