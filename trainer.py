import torch
from torch.autograd import Variable

from util.plotting import Plotting


class Trainer:

    def __init__(self, model, input_channels, seq_length, device, train_loader, test_loader, data_function,
                 optimizer, loss_fn, log_interval, batch_size,  permutetasion=None, do_plot=False, file_h=None):
        self.batch_size = batch_size
        self.steps = 0
        self.permutetasion = permutetasion
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.data_function = data_function
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.log_interval = log_interval
        self.batch_loss = []
        self.file_h = file_h
        self.do_plot = do_plot

    def reset_batch_loss(self):
        self.batch_loss = []

    def set_file_h(self, file_h):
        self.file_h = file_h

    def train(self, epoc):
        train_loss = 0
        self.model.train()
        for idx, (data, target) in enumerate(self.train_loader):

            data = data.to(self.device)
            target = target.to(self.device)

            data = self.data_function(data, self.permutetasion, self.input_channels, self.seq_length)

            output = self.model(data)
            loss = self.loss_fn(output, target)
            self.optimizer.zero_grad()
            loss.backward()

            train_loss += loss

            self.optimizer.step()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)

            self.steps += self.seq_length
            if idx > 0 and idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    epoc, idx * self.batch_size, len(self.train_loader.dataset),
                          100. * idx / len(self.train_loader), train_loss.item() / self.log_interval, self.steps))

                self.batch_loss.append(train_loss.item() / self.log_interval)
                if self.do_plot:
                    Plotting.plot_loss(self.batch_loss, self.log_interval)
                if self.file_h is not None:
                    self.file_h.write(epoc, self.steps, idx * self.batch_size, train_loss.item() / self.log_interval)

                train_loss = 0

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                data = self.data_function(data, self.permutetasion, self.input_channels, self.seq_length)

                output = self.model(data)
                test_loss += self.loss_fn(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(self.test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))

            self.file_h.write("TEST", test_loss, correct, len(self.test_loader.dataset),
                         100. * correct / len(self.test_loader.dataset))
            return test_loss

