import numpy as np
import math


def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x - max_x)/ np.sum(np.exp(x- max_x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

'''
configration include:
1. dimension of x 
2. length of recurrence layers
'''

class Single_Layer_LSTM():
    def __init__(self, dim_in, dim_out, dim_hide = 64, batch = 1):
            ''' 
            * dim_in: the number of features of x x
            * dim_hide: the number of features of hidden layer
            * dim_out: the number of feature of output layer
            * batch: number of batch
            '''
            self.dim_in = dim_in
            self.dim_out = dim_out
            self.dim_hide = dim_hide
            self.batch = batch
            self.whi, self.wxi, self.bi = _init_wh_wx_b() #输入门参数
            self.whf, self.wxf, self.bf = _init_wh_wx_b() #遗忘门参数
            self.who, self.wxo, self.bo = _init_wh_wx_b() #输出门参数
            self.wha, self.wxa, self.ba = _init_wh_wx_b() #输入新信息
            self.wy, self.by = np.random.uniform(-np.sqrt(1.0/self.dim_hide), np.sqrt(1.0/self.dim_hide), (self.dim_out, self.dim_hide)), \
            np.random.uniform(-np.sqrt(1.0/self.dim_hide), np.sqrt(1.0/self.dim_hide), (self.dim_out, batch))

    def _init_wh_wx_b(self):
        wh = np.random.uniform(-np.sqrt(1.0/self.dim_hide), np.sqrt(1.0/self.dim_hide), (self.dim_hide, self.dim_hide))
        wx = np.random.uniform(-np.sqrt(1.0/self.dim_hide), np.sqrt(1.0/self.dim_hide), (self.dim_hide, self.dim_in))
        b = np.random.uniform(-np.sqrt(1.0/self.dim_hide), np.sqrt(1.0/self.dim_hide), (self.dim_hide, self.batch))
        return wh, wx, b
    
    def _init_state_variable(self, T): # T是LSTM的时间长度
        iss = np.zeros((T + 1, self.dim_hide, self.batch)) #state of i_t
        fss = np.zeros((T + 1, self.dim_hide, self.batch)) #state of f_t
        ass = np.zeros((T + 1, self.dim_hide, self.batch)) #state of a_t
        oss = np.zeros((T + 1, self.dim_hide, self.batch)) #state of o_t
        hss = np.zeros((T + 1, self.dim_hide, self.batch)) #state of h_t
        css = np.zeros((T + 1, self.dim_hide, self.batch)) #state of c_t
        yss = np.zeros((T, self.dim_hide, self.batch)) #state of y_t

        # 这里除了yss之外的shape[0]都是T+1的原因是需要在该段时间之前的状态变量，即t = -1时的状态变量，这里初始状态为0
        return {'iss': iss, 'fss': fss, 'oss': oss, 'ass': ass, 'hss': hss, 'css': css, 'ys': ys}

    def forward(x):
        T = x.shape[0]
        stats = self._init_state_variable()
        for t in range(T): 
            # 前一时刻隐藏状态
            ht_pre = np.array(stats['hss'][t - 1])

            # calculate state variable
            stats['iss'][t] = self.cal_gate(self.whi, self.wxi, self.bi, ht_pre, x[t], sigmoid)
            stats['fss'][t] = self.cal_gate(self.whf, self.wxf, self.bf, ht_pre, x[t], sigmoid)
            stats['ass'][t] = self.cal_gate(self.wha, self.wxa, self.ba, ht_pre, x[t], tanh)
            stats['oss'][t] = self.cal_gate(self.who, self.wxo, self.bo, ht_pre, x[t], sigmoid)
            
            stats['css'][t] = stats['fss'][t] * stats['css'][t-1] + stats['iss'][t] * stats['ass'][t]
            stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t]) 

            stats['yss'][t] = softmax(np.dot(self.wy, stats['hss'][t]) + self.by)


    def cal_gate(self, wh, wx, b, ht_pre, x, activation): #activate 是激活函数
        return activation(np.matmul(wx, x) + np.matmul(wh, ht_pre) + b)
    
    def loss(self, x, y):
        stats = self.forward(x)
        return np.sum(np.square(stats['yss']-y)) / self.batch

    def _init_wh_wx_b_grad(self):
        dwh = np.zeros(self.whi.shape)
        dwx = np.zeros(self.wxi.shape)
        db = np.zeros(self.bi.shape)
        return dwh, dwx, db

    def back_propagation(self, x, y):
        dwhi, dwxi, dbi = self._init_wh_wx_b_grad()
        dwhf, dwxf, dbf = self._init_wh_wx_b_grad()
        dwha, dwxa, dba = self._init_wh_wx_b_grad()
        dwho, dwxo, dbo = self._init_wh_wx_b_grad()
        dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)
        # 初始化 delta_ct，因为后向传播过程中，此值需要累加
        delta_ct = np.zeros((self.dim_hide, 1))

        # 前向计算
        stats = self.forward(x)
        # 目标函数对输出 y 的偏导数
        delta_o = stats['yss']
        delta_o[np.arange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            # yt = softmax(self.wy.dot(ht) + self.by)
            # 输出层wy, by的偏导数，由于所有时刻的输出共享输出权值矩阵，故所有时刻累加
            dwy += delta_o[t].dot(stats['hss'][t].reshape(1, -1))  
            dby += delta_o[t]

            # 目标函数对隐藏状态的偏导数
            delta_ht = self.wy.T.dot(delta_o[t])

            # 各个门及状态单元的偏导数
            delta_ot = delta_ht * tanh(stats['css'][t])
            delta_ct += delta_ht * stats['oss'][t] * (1-tanh(stats['css'][t])**2)
            delta_it = delta_ct * stats['ass'][t]
            delta_ft = delta_ct * stats['css'][t-1]
            delta_at = delta_ct * stats['iss'][t]

            delta_at_net = delta_at * (1-stats['ass'][t]**2)
            delta_it_net = delta_it * stats['iss'][t] * (1-stats['iss'][t])
            delta_ft_net = delta_ft * stats['fss'][t] * (1-stats['fss'][t])
            delta_ot_net = delta_ot * stats['oss'][t] * (1-stats['oss'][t])

            # 更新各权重矩阵的偏导数，由于所有时刻共享权值，故所有时刻累加
            dwhf, dwxf, dbf = self._cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t-1], x[t])                              
            dwhi, dwxi, dbi = self._cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t-1], x[t])                              
            dwha, dwxa, dba = self._cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t-1], x[t])            
            dwho, dwxo, dbo = self._cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t-1], x[t])

        return [dwhf, dwxf, dbf, 
                dwhi, dwxi, dbi, 
                dwha, dwxa, dba, 
                dwho, dwxo, dbo, 
                dwy, dby]

      # 更新各权重矩阵的偏导数            
    def _cal_grad_delta(self, dwh, dwx, db, delta_net, ht_pre, x):
        dwh += delta_net * ht_pre
        dwx += delta_net * x
        db += delta_net

        return dwh, dwx, db

    # 计算梯度, (x,y)为一个样本
    def sgd_step(self, x, y, learning_rate):
        dwhf, dwxf, dbf, \
        dwhi, dwxi, dbi, \
        dwha, dwxa, dba, \
        dwho, dwxo, dbo, \
        dwy, dby = self.back_propagation(x, y)

        # 更新权重矩阵
        self.whf, self.wxf, self.bf = self._update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
        self.whi, self.wxi, self.bi = self._update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi, dwxi, dbi)
        self.wha, self.wxa, self.ba = self._update_wh_wx(learning_rate, self.wha, self.wxa, self.ba, dwha, dwxa, dba)
        self.who, self.wxo, self.bo = self._update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho, dwxo, dbo)

        self.wy, self.by = self.wy - learning_rate * dwy, self.by - learning_rate * dby

    # 更新权重矩阵
    def _update_wh_wx(self, learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate * dwh
        wx -= learning_rate * dwx
        b -= learning_rate * db

        return wh, wx, b

    # 训练 LSTM
    def train(self, X_train, y_train, learning_rate=0.005, n_epoch=5):
        #X_train (seq, dim_in, batch)
        #Y_train (seq, dim_out, batch)
        losses = []
        num_examples = 0

        for epoch in xrange(n_epoch):   
            for i in xrange(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples += 1

            loss = self.loss(X_train, y_train)
            losses.append(loss)
            print 'epoch {0}: loss = {1}'.format(epoch+1, loss)
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                print 'decrease learning_rate to', learning_rate