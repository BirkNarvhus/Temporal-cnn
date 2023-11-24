import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.data import data_generator
from util.fileHandeling import FileHandeling


class Attention(nn.Module):
    def __init__(self, device, hidden_size):
        super(Attention, self).__init__()
        self.device = device

        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size, hidden_size)
        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, rnn_outputs, final_hidden_state):
        seq_len, batch_size, _ = rnn_outputs.shape
        rnn_outputs = rnn_outputs.transpose(0, 1)
        attn_weights = self.attn(rnn_outputs)  # (batch_size, seq_len, hidden_dim)
        attn_weights = torch.bmm(attn_weights, final_hidden_state.unsqueeze(2))

        attn_weights = F.softmax(attn_weights.squeeze(2), dim=1)

        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(2)
        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state), dim=1)))

        return attn_hidden, attn_weights


class RnnClassifier(nn.Module):
    def __init__(self, device, input_size, rnn_hidden_dim, num_layers, dropout, linear_dims, label_size):
        super(RnnClassifier, self).__init__()
        self.device = device

        # Embedding layer

        # Calculate number of directions
        self.num_directions = 2
        self.num_layers = num_layers
        self.rnn_hidden_dim = rnn_hidden_dim

        self.linear_dims = [rnn_hidden_dim * self.num_directions] + linear_dims
        self.linear_dims.append(label_size)

        self.lstm = nn.LSTM(input_size,
                            rnn_hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)

        # Define set of fully connected layers (Linear Layer + Activation Layer) * #layers
        self.linears = nn.ModuleList()
        for i in range(0, len(self.linear_dims) - 1):
            if dropout > 0.0:
                self.linears.append(nn.Dropout(p=dropout))
            linear_layer = nn.Linear(self.linear_dims[i], self.linear_dims[i + 1])
            self.init_weights(linear_layer)
            self.linears.append(linear_layer)
            if i == len(self.linear_dims) - 1:
                break  # no activation after output layer!!!
            self.linears.append(nn.ReLU())

        self.hidden = None

        self.attn = Attention(self.device, rnn_hidden_dim * self.num_directions)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.rnn_hidden_dim).to(
                self.device),
            torch.zeros(self.num_layers * self.num_directions, batch_size, self.rnn_hidden_dim).to(
                self.device))

    def freeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        batch_size, seq_len, inn = inputs.shape

        # Push through RNN layer
        rnn_output, self.hidden = self.lstm(inputs, self.hidden)

        final_state = \
            self.hidden[0].view(inn, self.num_layers, self.num_directions, batch_size,
                                self.rnn_hidden_dim)[:, -1]
        final_state = final_state.squeeze()
        # Handle directions
        final_hidden_state = None

        h_1, h_2 = final_state[0], final_state[1]
        final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        # Push through attention layer
        attn_weights = None
        rnn_output = rnn_output.permute(1, 0, 2)  #
        X, attn_weights = self.attn(rnn_output, final_hidden_state)

        # Push through linear layers
        for l in self.linears:
            X = l(X)

        log_probs = F.log_softmax(X, dim=1)

        return log_probs, attn_weights

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            print("Initialize layer with nn.init.xavier_uniform_: {}".format(layer))
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

root = '../data/mnist'


def test():
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for idx, (inputs, target) in enumerate(test_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            inputs = inputs.view(-1, input_size, sequence_length)
            inputs = inputs.transpose(1, 2)

            bsize, _, _ = inputs.shape

            if permute:
                inputs = inputs[:, permutetasion, :]

            model.hidden = model.init_hidden(bsize)

            outputs, weights_test = model(inputs)
            curr_loss = loss_fn(outputs, target)
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            # Validation
            test_loss += curr_loss.item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        if file_h is not None:
            file_h.write("TEST", test_loss, correct, len(test_loader.dataset),
                            100. * correct / len(test_loader.dataset))


file_h = None

if __name__ == '__main__':



    n_epochs = 20
    input_size = 1
    sequence_length = int(784 / input_size)
    log_interval = 20
    batchsize = 32
    lr = 0.0001
    steps = 0
    permute = True
    permutetasion = torch.Tensor(np.random.permutation(784).astype(np.float64)).long().to(device)

    hidden_size = 100
    num_layers = 1
    dropout = 0.05

    model = RnnClassifier(device, input_size, hidden_size, num_layers, dropout, [], 10)
    model.to(device)

    do_write = True

    if do_write:
        filename = 'LSTMATTN-hs{}-l{}-d{}-{}'.format(hidden_size, num_layers, dropout, "P" if permute else "nP")
        file_h = FileHandeling(file_name=filename, sub_folder="lstmattn", root_default="../result/")

    train_loader, test_loader = data_generator(root, batchsize)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    batch_loss = []
    for epoch in range(n_epochs):
        train_loss = 0
        model.train()
        for idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_batch = X_batch.view(-1, input_size, sequence_length)
            X_batch = X_batch.transpose(1, 2)
            bs, _, _ = X_batch.shape

            if permute:
                X_batch = X_batch[:, permutetasion, :]

            model.hidden = model.init_hidden(bs)

            y_pred, weights = model(X_batch)

            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Validation
            train_loss += loss
            steps += bs
            if idx > 0 and idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    epoch, idx * bs, len(train_loader.dataset),
                           100. * idx / len(train_loader), train_loss.item() / log_interval, steps))

                batch_loss.append(train_loss.item() / log_interval)

                if file_h is not None:
                    file_h.write(epoch, steps, idx * bs, train_loss.item() / log_interval)
                train_loss = 0
        test()

        if file_h is not None:
            file_h.flush()

        if epoch % 10 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Number of params: {}'.format(pytorch_total_params))

    file_h.write(pytorch_total_params)
