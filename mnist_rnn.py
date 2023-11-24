from util.data import data_generator
import torch.nn as nn
import torch
from models.rnn import Rnn
import numpy as np
from util.plotting import Plotting
import matplotlib.pyplot as plt
from trainer import Trainer
from util.fileHandeling import FileHandeling

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"


root = './data/mnist'
batch_size = 100
log_interval = 5
lr = 0.001
input_size = 1
sequence_length = int(784 / input_size)
n_epochs = 20
steps = 0
permute = False
permutetasion = torch.Tensor(np.random.permutation(784).astype(np.float64)).long().to(device)
hidden_size = 128
num_layers = 4
dropout = 0.05
do_plot = False

train_loader, test_loader = data_generator(root, batch_size)

model = Rnn(input_size, hidden_size, num_layers, dropout, 10, device)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)
loss_fn = nn.functional.nll_loss


def data_fun(data, perm, input_channels, seq_length):
    data = data.view(-1, seq_length, input_channels)
    if perm is not None:
        data = data[:, perm, :]
    
    return data


trainer = Trainer(model, input_size, sequence_length, device, train_loader, test_loader, data_fun, optimizer,
                  loss_fn, log_interval, batch_size, permutetasion if permute else None, do_plot)


if __name__ == '__main__':
    filename = 'RNN-hs{}-l{}-d{}-{}'.format(hidden_size, num_layers, dropout, "P" if permute else "nP")
    file_h = FileHandeling(file_name=filename, sub_folder="rnn")

    trainer.set_file_h(file_h)

    for epoch in range(1, n_epochs + 1):
        trainer.reset_batch_loss()

        if do_plot:
            plt.clf()

        trainer.train(epoch)

        if do_plot:
            Plotting.plot_loss(trainer.batch_loss, log_interval, True)

        trainer.test()

        if file_h is not None:
            file_h.flush()

        if do_plot:
            plt.show()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Number of params: {}'.format(pytorch_total_params))

    file_h.write(pytorch_total_params)
