import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

net = Net()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)

trainingdataX = [[[0.01, 0.01], [0.01, 0.90], [0.90, 0.01], [0.95, 0.95]], [[0.02, 0.03], [0.04, 0.95], [0.97, 0.02], [0.96, 0.95]]]
trainingdataY = [[[0.01], [0.90], [0.90], [0.01]], [[0.04], [0.97], [0.98], [0.1]]]

n_epochs = 20000

for epoch in range(n_epochs):

    for i, data in enumerate(trainingdataX, 0):
        inputs = data
        labels = trainingdataY[i]
        inputs = Variable(torch.FloatTensor(inputs))
        labels = Variable(torch.FloatTensor(labels))
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if(i%5000==0):
            print(str(loss.data[0]))

print("Training done")
print(net(Variable(torch.FloatTensor(trainingdataX[0]))))
