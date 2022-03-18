import numpy as np
import gzip,urllib.request,struct
import os.path
import tqdm
def gen_spiral_dataset(N=100,D=2,K=3):
    X = np.zeros((N*K,D))
    y = np.zeros(N*K,dtype='uint8')
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N)
        t = np.linspace(j*4,(j+1)*4,N)
        X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
        y[ix] = j
    return X,y

def gen_mnist_dataset():
    def load_images(buffers):
        _,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
        bits = num * rows * cols
        images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
        images = np.reshape(images, [num, rows * cols])
        return images
    def load_labels(buffers):
        _,num = struct.unpack_from('>II', buffers, 0) 
        labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
        labels = np.reshape(labels, [num])
        return labels
    url = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
           'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
           'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
           'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    filenames = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for i in range(len(filenames)):
        if not os.path.isfile(filenames[i]):
            print("download dataset",filenames[i])
            urllib.request.urlretrieve(url[i], filenames[i])

    with gzip.open("train-images-idx3-ubyte.gz","rb") as f:
        train_image = load_images(f.read())
    with gzip.open("train-labels-idx1-ubyte.gz","rb") as f:
        train_label = load_labels(f.read())
    with gzip.open("t10k-images-idx3-ubyte.gz","rb") as f:
        test_image = load_images(f.read())
    with gzip.open("t10k-labels-idx1-ubyte.gz","rb") as f:
        test_label = load_labels(f.read())
    return (train_image,train_label),(test_image,test_label)

def gen_fashion_dataset():
    def load_images(buffers):
        _,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
        bits = num * rows * cols
        images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
        images = np.reshape(images, [num, rows * cols])
        return images
    def load_labels(buffers):
        _,num = struct.unpack_from('>II', buffers, 0) 
        labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
        labels = np.reshape(labels, [num])
        return labels
    url = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
           'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
           'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
           'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz']
    filenames = ['fashion-train-images-idx3-ubyte.gz','fashion-train-labels-idx1-ubyte.gz','fashion-t10k-images-idx3-ubyte.gz','fashion-t10k-labels-idx1-ubyte.gz']
    for i in range(len(filenames)):
        if not os.path.isfile(filenames[i]):
            print("download dataset",filenames[i])
            urllib.request.urlretrieve(url[i], filenames[i])

    with gzip.open("fashion-train-images-idx3-ubyte.gz","rb") as f:
        train_image = load_images(f.read())
    with gzip.open("fashion-train-labels-idx1-ubyte.gz","rb") as f:
        train_label = load_labels(f.read())
    with gzip.open("fashion-t10k-images-idx3-ubyte.gz","rb") as f:
        test_image = load_images(f.read())
    with gzip.open("fashion-t10k-labels-idx1-ubyte.gz","rb") as f:
        test_label = load_labels(f.read())
    return (train_image,train_label),(test_image,test_label)
