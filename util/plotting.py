import torch
import matplotlib.pyplot as plt
import numpy as np


class Plotting:
    @staticmethod
    def plot_loss(batch_loss, log_interval, show_result=False):
        plt.figure(1)
        loss_t = torch.tensor(batch_loss, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.plot(np.linspace(0, len(loss_t) * log_interval, len(loss_t)), loss_t.numpy())
        if len(loss_t) >= 10:
            means = loss_t.unfold(0, 10, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(9), means))
            plt.plot(np.linspace(0, len(means) * log_interval, len(means)), means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are update
