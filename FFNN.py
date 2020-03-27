import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import torch.nn.functional as F


class FFNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, non_lin, output_size=1, lr=0.01):
        super(FFNNModel, self).__init__()
        self.hidden_size = hidden_size
        activation = {"relu": nn.ReLU(), "tanh": nn.Tanh()}
        self.activ = non_lin
        if self.activ:
            self.non_lin = activation[non_lin]
        if self.hidden_size > 0:
            self.linear0 = nn.Linear(input_size, self.hidden_size)
            self.linear1 = nn.Linear(self.hidden_size, output_size)
        else:
            self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.lr = lr

    def forward(self, x):
        if self.hidden_size == 0:
            n = self.linear(x)
        else:
            if self.activ:
                l = self.linear0(x)
                m = self.non_lin(l)
                n = self.linear1(m)
            else:
                m = self.linear0(x)
                n = self.linear1(m)
        o = self.sigmoid(n)
        return o

    def train_model(self, samples, epoch=3):
        for epoch in range(epoch):
            criterion = nn.BCELoss()
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
            for sample in samples:
                x_vector, y_label = sample[0], sample[1]
                x_vector, y_label = Variable(torch.FloatTensor([x_vector]), requires_grad=True), Variable(torch.FloatTensor([[y_label]]))
                output = self.forward(x_vector)
                #print(output)
                loss = criterion(output, y_label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self, testdata, labels):
        list_pred = []
        i = 0
        for inputvector in testdata:
            prediction = self.forward(inputvector)
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0
            list_pred.append(prediction)
            i += 1
        accuracy = accuracy_score(labels, list_pred)
        f = f1_score(labels, list_pred, average='weighted')
        recall = recall_score(labels, list_pred, average='weighted')
        precision = precision_score(labels, list_pred, average='weighted')
        print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f)
        return accuracy, precision, recall, f



