import numpy as np
import torch.nn.functional as F
from util.data import data_generator
from models.tcnModel import TcnModel
from util.plotting import Plotting
import torch
import matplotlib.pyplot as plt
from util.fileHandeling import FileHandeling
from trainer import Trainer

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"

root = './data/mnist'
batch_size = 64
dropout = 0.05
n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = 5
steps = 0
levels = 8
hidden_units_pr_layer = 25
channel_sizes = [hidden_units_pr_layer] * levels
kernel_size = 7
learning_rate = 2e-3
log_interval = 10
do_plot = False
permute = False

permutetasion = torch.Tensor(np.random.permutation(784).astype(np.float64)).long().to(device)

train_loader, test_loader = data_generator(root, batch_size)
model = TcnModel(input_channels, n_classes, channel_sizes, kernel_size, dropout)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), learning_rate)


file_h = None


def data_fun(data, perm, input_channels, seq_length):
    data = data.view(-1, input_channels, seq_length)
    if perm is not None:
        data = data[:, :, perm]
    return data


trainer = Trainer(model, input_channels, seq_length, device, train_loader, test_loader, data_fun, optimizer,
                      F.nll_loss, log_interval, batch_size, permutetasion if permute else None, do_plot)

if __name__ == "__main__":
    filename = 'TCN-hul{}-l{}-d{}-{}'.format(hidden_units_pr_layer, levels, dropout, "P" if permute else "nP")
    file_h = FileHandeling(file_name=filename, sub_folder="tcn")

    trainer.set_file_h(file_h)
    for epoch in range(1, epochs+1):
        trainer.reset_batch_loss()

        if do_plot:
            plt.clf()

        trainer.train(epoch)

        if do_plot:
            Plotting.plot_loss(trainer.batch_loss, log_interval, True)

        trainer.test()

        file_h.flush()

        if do_plot:
            plt.show()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Number of params: {}'.format(pytorch_total_params))

    file_h.write(pytorch_total_params)




