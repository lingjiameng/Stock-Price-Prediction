import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
import pandas as pd

# Hyper Parameters
# train the training data n times, to save time, we just train 1 epoch
EPOCH = 50
BATCH_SIZE = 32
TIME_STEP = 10          # rnn time step / image height
INPUT_SIZE = 8         # rnn input size / image width
LR = 1e-4               # learning rate
DATASIZE = 427700


def csv2train(filename):
    df = pd.read_csv(filename, index_col=0)
    groups = df.groupby('Date')

    train_data = []
    train_tag = []
    for date, group in groups:
        print("load data for date:", date)
        # cal diff of volume
        col_name = group.columns.tolist()
        col_name.insert(col_name.index("Volume"), "VolDiff")
        group = group.reindex(columns=col_name)
        # print(group.columns)

        group.loc[:, ["VolDiff"]] = group["Volume"] - group["Volume"].shift(1)
        group_new = group.fillna(0)
        # print(group_new.columns)

        # trans to numpy
        dataset = np.array(group_new.loc[:, ['MidPrice', 'LastPrice', 'Volume',
                                             'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'VolDiff']])

        # norm data in size of group
        dataset.astype(np.float32)
        mean = dataset.mean(axis=0)
        std = dataset.std(axis=0)
        # print("mean:\n",mean,'\nstd:\n',std)
        dataset = (dataset - mean)/std

        #add to data set with dropout improper additional data
        size = len(dataset)-30
        size = size - (size % BATCH_SIZE)
        for i in range(0, size, 1):
            train_data.append(dataset[i:i+10])
            train_tag.append(dataset[i+10:i+30, 0].mean())
    train_data = torch.FloatTensor(train_data)
    train_tag = torch.FloatTensor(train_tag)

    return (train_data,train_tag), 0, 1


def csv2test(filename):
    df = pd.read_csv(filename, index_col=0)
    groups = df.groupby('Date')

    test_data = []
    test_data_means = []
    test_data_stds = []
    for date, group in groups:
        # print("load data for date:", date)
        # cal diff of volume
        col_name = group.columns.tolist()
        col_name.insert(col_name.index("Volume"), "VolDiff")
        group = group.reindex(columns=col_name)
        # print(group.columns)
        group["VolDiff"] = group["Volume"] - group["Volume"].shift(1)
        group_new = group.fillna(0)
        # print(group_new.columns)

        # trans to numpy
        dataset = np.array(group_new.loc[:, ['MidPrice', 'LastPrice', 'Volume',
                                             'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'VolDiff']])

        # norm data in size of group
        dataset.astype(np.float32)
        dataset.astype(np.float32)
        mean = dataset.mean(axis=0)
        std = dataset.std(axis=0)
        # print("mean:\n", mean, '\nstd:\n', std)
        dataset = (dataset - mean)/std

        mean_midprice = mean[0]
        std_midprice = std[0]

        #add to data set with dropout improper additional data
        size = len(dataset)
        for i in range(0, size, 10):
            test_data.append(dataset[i:i+10])
            test_data_means.append(mean_midprice)
            test_data_stds.append(std_midprice)

    return test_data, test_data_means, test_data_stds


def res2csv(dataset):
    fout = open("test_predict.csv", "w")
    fout.write("caseid,midprice\n")
    for i in range(len(dataset)):
        if(i < 142):
            continue
        fout.write("%d,%f\n" % (i+1, dataset[i]))
    fout.close()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # None represents zero initial hidden state
        r_out, (h_n, h_c) = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


train_data, train_mean, train_std = csv2train("data/train_data.csv")
print(len(train_data[0]))

train_dataset = Data.TensorDataset(train_data[0],train_data[1])
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

rnn = RNN()
rnn.cuda()
print(rnn)
# optimize all cnn parameters
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, betas=(
    0.9, 0.999), eps=1e-08, weight_decay=0.01)
# the target label is not one-hotted
loss_func = nn.MSELoss()
# training and testing

for epoch in range(EPOCH):
    avg_loss = 0.0
    count = 1
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data
        # reshape x to (batch, time_step, input_size)
        # print(b_x, b_y)
        # b_x = b_x.view(-1, 10, INPUT_SIZE).float()
        #print(b_x)
        # b_y = b_y.float()

        b_x = Variable(b_x).cuda()
        b_y = Variable(b_y).cuda()

        output = rnn(b_x)                               # rnn output
        # cross entropy loss
        loss = loss_func(output.view(BATCH_SIZE), b_y)
        # clear gradients for this training step
        optimizer.zero_grad()
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients
        # print(loss.data)
        avg_loss += loss.data.item()
        count += 1
        # if step % 50 == 0:
    print('Epoch: ', epoch, '| train loss: %.8f' % (avg_loss/count))

#trans to cpu
rnn.cpu()

#load test data
test_data, test_data_means, test_data_stds = csv2test("data/test_data.csv")

#predict
test_data = torch.tensor(test_data).float()
test_output = rnn(test_data)
test_output = test_output.detach().numpy().reshape(-1)

test_data_means = np.array(test_data_means)
test_data_stds = np.array(test_data_stds)

# print(test_output.shape)
# print(test_data_means.shape)
# print(test_data_stds.shape)
#re scaler
test_output = (test_output*test_data_stds)+test_data_means

print(test_output.shape)
res2csv(test_output)
