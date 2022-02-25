import numpy as np
def onehot(label,class_num):
    one_hot = np.zeros((label.shape[0],class_num))
    one_hot[np.arange(label.shape[0]),label] = 1
    return one_hot