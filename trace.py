#!/usr/bin/python
import ConfigParser
import numpyp

import os.path
import pickle
import os
from collections import OrderedDict
import time

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

global id_count
id_count = 0

def print_id(x):#设置addresses-locality.txt
    global id_count
    id_count += 1
    output = open('addresses-locality.txt','a')
    print(x, file = output)
    output.close()
    print(id_count)
    if(id_count == 10000):
        input('已写入10000个地址，此时停止程序')
    
    
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    print_id(id(T))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        print_id(id(row[X[idx]]))
        
    return T

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    with open(save_file, 'rb') as f:
        print_id(id(save_file))
        dataset = pickle.load(f)
        print_id(id(dataset))
    
    print_id(id(normalize))
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            print_id(id(dataset[key]))
            dataset[key] /= 255.0
            print_id(id(dataset[key]))
    
    print_id(id(one_hot_label))       
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        print_id(id(dataset['train_label']))
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
        print_id(id(dataset['test_label']))
    
    print_id(id(flatten))
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
            print_id(id(dataset[key]))

    return (dataset['train_img'], dataset['train_label']), \
            (dataset['test_img'], dataset['test_label']) 

class Relu:
    def __init__(self):
        self.mask = None
        print_id(id(self.mask))

    def forward(self, x):
        self.mask = (x <= 0)
        print_id(id(self.mask))
        out = x.copy()
        print_id(id(out))
        out[self.mask] = 0
        print_id(id(out[self.mask]))

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        print_id(id(dout[self.mask]))
        dx = dout
        print_id(id(dx))

        return dx

class FC:
    def __init__(self, W, b):
        self.W =W
        print_id(id(self.W))
        self.b = b
        print_id(id(self.b))
        
        self.x = None
        print_id(id(self.x))
        self.original_x_shape = None
        print_id(id(self.original_x_shape))
        # 权重和偏置参数的导数
        self.dW = None
        print_id(id(self.dW))
        self.db = None
        print_id(id(self.db))

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        print_id(id(self.original_x_shape))
        x = x.reshape(x.shape[0], -1)
        print_id(id(x))
        self.x = x
        print_id(id(self.x))

        out = np.dot(self.x, self.W) + self.b
        print_id(id(out))

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        print_id(id(dx))
        self.dW = np.dot(self.x.T, dout)
        print_id(id(self.dW))
        self.db = np.sum(dout, axis=0)
        print_id(id(self.db))
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        print_id(id(dx))
        return dx

def softmax(x):
    print_id(id(x.ndim))
    if x.ndim == 2:
        x = x.T
        print_id(id(x))
        x = x - np.max(x, axis=0)
        print_id(id(x))
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        print_id(id(y))
        return y.T 

    x = x - np.max(x) # 溢出对策
    print_id(id(x))
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    print_id(id(y.ndim))
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        print_id(id(t))
        y = y.reshape(1, y.size)
        print_id(id(y))
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    print_id(id(t.size))
    print_id(id(y.size))
    if t.size == y.size:
        t = t.argmax(axis=1)
        print_id(id(t))
             
    batch_size = y.shape[0]
    print_id(id(batch_size))
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        print_id(id(self.loss))
        self.y = None # softmax的输出
        print_id(id(self.y))
        self.t = None # 监督数据
        print_id(id(self.t))

    def forward(self, x, t):
        self.t = t
        print_id(id(self.t))
        self.y = softmax(x)
        print_id(id(self.y))
        self.loss = cross_entropy_error(self.y, self.t)
        print_id(id(self.loss))
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        print_id(id(batch_size))
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
            print_id(id(dx))
        else:
            dx = self.y.copy()
            print_id(id(dx))
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    print_id(id(input_data.shape))
    out_h = (H + 2*pad - filter_h)//stride + 1
    print_id(id(out_h))
    out_w = (W + 2*pad - filter_w)//stride + 1
    print_id(id(out_w))

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    print_id(id(img))
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    print_id(id(col))

    for y in range(filter_h):
        y_max = y + stride*out_h
        print_id(id(y_max))
        for x in range(filter_w):
            x_max = x + stride*out_w
            print_id(id(x_max))
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            print_id(id(col[:, :, y, x, :, :]))

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    print_id(id(col))
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    print_id(id(out_h))
    
    out_w = (W + 2*pad - filter_w)//stride + 1
    print_id(id(out_w))
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    print_id(id(col))

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    print_id(id(img))
    for y in range(filter_h):
        y_max = y + stride * out_h
        print_id(id(y_max))
        for x in range(filter_w):
            x_max = x + stride*out_w
            print_id(id(x_max))
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
            print_id(id(img[:, :, y:y_max:stride, x:x_max:stride]))

    return img[:, :, pad:H + pad, pad:W + pad]

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        print_id(id(self.W))
        self.b = b
        print_id(id(self.b))
        self.stride = stride
        print_id(id(self.stride))
        self.pad = pad
        print_id(id(self.pad))
        
        # 中间数据（backward时使用）
        self.x = None 
        print_id(id(self.x))
        self.col = None
        print_id(id(self.col))
        self.col_W = None
        print_id(id(self.col_W))
        
        # 权重和偏置参数的梯度
        self.dW = None
        print_id(id(self.dW))
        self.db = None
        print_id(id(self.db))

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        print_id(id(out_h))
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
        print_id(id(out_w))

        col = im2col(x, FH, FW, self.stride, self.pad)
        print_id(id(col))
        col_W = self.W.reshape(FN, -1).T
        print_id(id(col_W))

        out = np.dot(col, col_W) + self.b
        print_id(id(out))
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        print_id(id(out))

        self.x = x
        print_id(id(self.x))
        self.col = col
        print_id(id(self.col))
        self.col_W = col_W
        print_id(id(self.col_W))

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)
        print_id(id(dout))

        self.db = np.sum(dout, axis=0)
        print_id(id(self.db))
        self.dW = np.dot(self.col.T, dout)
        print_id(id(self.dW))
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        print_id(id(self.dW))

        dcol = np.dot(dout, self.col_W.T)
        print_id(id(dcol))
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        print_id(id(dx))

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        print_id(id(self.pool_h))
        self.pool_w = pool_w
        print_id(id(self.pool_h))
        self.stride = stride
        print_id(id(self.stride))
        self.pad = pad
        print_id(id(self.pad))
        
        self.x = None
        print_id(id(self.x))
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        print_id(id(out_h))
        out_w = int(1 + (W - self.pool_w) / self.stride)
        print_id(id(out_w))

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        print_id(id(col))
        col = col.reshape(-1, self.pool_h*self.pool_w)
        print_id(id(col))

        arg_max = np.argmax(col, axis=1)
        print_id(id(arg_max))
        out = np.max(col, axis=1)
        print_id(id(out))
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        print_id(id(out))

        self.x = x
        print_id(id(self.x))
        self.arg_max = arg_max
        print_id(id(self.arg_max))

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        print_id(id(dout))
        
        pool_size = self.pool_h * self.pool_w
        print_id(id(pool_size))
        dmax = np.zeros((dout.size, pool_size))
        print_id(id(dmax))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        print_id(id(dcol))
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        print_id(id(dx))
        return dx

class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        print_id(id(self.dropout_ratio))
        self.mask = None
        print_id(id(self.mask))

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            print_id(id(self.mask))
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class ConvNet:  
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param_1 = {'filter_num':30, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=50, output_size=10):
        
        # 初始化权重===========
        # 各层的神经元平均与前一层的几个神经元有连接
        pre_node_nums = np.array([1*3*3, 30*3*3, 3136, hidden_size])
        print_id(id(pre_node_nums))
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # 使用He初始值
        print_id(id(weight_init_scales))
        
        self.params = {}
        print_id(id(self.params))
        pre_channel_num = input_dim[0]
        print_id(id(pre_channel_num))
        for idx, conv_param in enumerate([conv_param_1, conv_param_2]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn\
            (conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], \
             conv_param['filter_size'])
            print_id(id(self.params['W' + str(idx+1)]))
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            print_id(id(self.params['b' + str(idx+1)]))
            pre_channel_num = conv_param['filter_num']
            print_id(id(pre_channel_num))
        self.params['W3'] = weight_init_scales[2] * np.random.randn(3136, hidden_size)
        self.params['b3'] = np.zeros(hidden_size)
        self.params['W4'] = weight_init_scales[3] * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)
        print_id(id(self.params['W3']))
        print_id(id(self.params['b3']))
        print_id(id(self.params['W4']))
        print_id(id(self.params['b4']))

        input_size = input_dim[1]
        print_id(id(input_size))
        '''conv_output_size_1 = (input_size - conv_param_1['filter_size'] + 2*conv_param_1['pad']) / conv_param_1['stride'] + 1
        pool_output_size_1 = int(conv_param_1['filter_num'] * (conv_output_size_1/2) * (conv_output_size_1/2))
        conv_output_size_2 = (conv_param_1['filter_num'] - conv_param_2['filter_size'] + 2*conv_param_2['pad']) / conv_param_2['stride'] + 1
        pool_output_size_2 = int(conv_param_2['filter_num'] * (conv_output_size_2/2) * (conv_output_size_2/2))
        '''
        
        
        # 生成层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        self.layers['FC1'] = FC(self.params['W3'], self.params['b3'])
        self.layers['Relu2'] = Relu()
        self.layers['Dropout1'] = Dropout(0.1)
        self.layers['FC2'] = FC(self.params['W4'], self.params['b4'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        """求梯度（误差反向传播法）

        Parameters
        ----------
        x : 输入数据
        t : 正确标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['FC1'].dW, self.layers['FC1'].db
        grads['W4'], grads['b4'] = self.layers['FC2'].dW, self.layers['FC2'].db

        return grads
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]    
    
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'FC1', 'FC2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
            
            

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs, mini_batch_size,
                 optimizer='Adam', optimizer_param={'lr':0.01}):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size

        # optimizer
        optimizer_class_dict = {'Adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 1
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        for i in range(self.max_iter):
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            print_id(id(batch_mask))
            print_id(id(x_batch))
            print_id(id(t_batch))
            
            grads = self.network.gradient(x_batch, t_batch)
            print_id(id(grads))
            self.optimizer.update(self.network.params, grads)
            
            loss = self.network.loss(x_batch, t_batch)
            print_id(id(loss))
            self.current_iter += 1
            print_id(id(self.current_iter))
            
            if (self.current_iter - 1) % 60 == 0:
                self.train_loss_list.append(loss)
                print_id(id(self.train_loss_list))
                #print_id("epoch:%d" %self.current_epoch, end = ' ')
                iter_count = self.current_iter % self.iter_per_epoch
                print_id(id(iter_count))
                #print_id("iter:%d" %iter_count, end = ' ')
                elapsed = (time.perf_counter() - start)
                print_id(id(elapsed))
                #print_id("Time used:%.2f" %elapsed, end = ' ')
                #print_id("train loss:%.10f" %loss)
            
            if self.current_iter % self.iter_per_epoch == 0:
                train_acc = self.network.accuracy(x_train, t_train)
                print_id(id(train_acc))
                test_acc = self.network.accuracy(x_test, t_test)
                print_id(id(test_acc))
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
    
                #print_id("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
                self.current_epoch += 1
                print_id(id(self.current_epoch))

