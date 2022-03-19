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
    nn.Conv2d(1,6,5,1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.BatchNorm2d(6),
    nn.Conv2d(6,16,5,1),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.BatchNorm2d(16),
    nn.flatten(),
    nn.Linear(120),
    nn.ReLU(),
    nn.Linear(84),
    nn.ReLU(),
    nn.Linear(10),
    nn.Softmax(cross_entropy=True)
])

def train(f,x,y,lr=0.001,epochs=500,batch_size=256):
    loss_list = []
    loss_func = nn.CrossEntropy()
    optimizer = optim.Adam(lr)
    for _ in tqdm.tqdm(range(epochs)):
        f.train()
        # utils.gradient_check(f,x,y,loss_func)
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
        f.eval()
        test = Tensor(test_x.reshape(-1,1,28,28).astype(np.float32)/255,requires_grad=False)
        print("acc",np.mean(np.argmax(f(test).data,axis=1)==test_y))

    return loss_list

epochs = 10
(train_x,train_y),(test_x,test_y) = td.gen_mnist_dataset()
loss_list = train(f,train_x,train_y,lr=0.001,epochs=epochs)
f.eval()
test_x = Tensor(test_x.reshape(-1,1,28,28).astype(np.float32)/255,requires_grad=False)
print("acc",np.mean(np.argmax(f(test_x).data,axis=1)==test_y))
y = np.array(loss_list)
x = np.arange(len(y))
plt.plot(x,y,color='red',label='loss')
plt.legend()
plt.show()
