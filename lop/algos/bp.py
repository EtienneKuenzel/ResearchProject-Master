import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

class Backprop(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0):
        self.net = net
        self.to_perturb = to_perturb
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]


    def learn(self, x, target):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target.long())

        loss.backward()
        self.opt.step()
        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()


class DQN_EWC_Policy(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0):
        self.net = net
        self.device = device
        self.lambda_ewc = 0.5  # Regularization strength for EWC
        self.ewc_penalty =0.0
        self.fisher_matrices = []  # List of Fisher matrices for each task
        self.prev_params_list = []  # List of previous parameters for each task

        # Define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                  weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)

        # Define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

    def compute_fisher_matrix(self, dataloader, dev='cpu'):
        """
        Compute the Fisher Information Matrix (FIM) for the current task using ImageNet data.
        :param dataloader: Tuple containing (x_train, y_train, x_test, y_test).
        :param dev: Device to use ('cpu' or 'cuda').
        """
        mini_batch_size = 1
        x_train, y_train, x_test, y_test = dataloader

        # Ensure inputs and labels are on the specified device
        x_train, y_train = x_train.float().to(dev), y_train.to(dev)
        x_test, y_test = x_test.float().to(dev), y_test.to(dev)

        # Initialize a Fisher matrix for the current task
        fisher_matrix = {name: torch.zeros_like(param, device=dev)
                         for name, param in self.net.named_parameters() if param.requires_grad}
        self.net.eval()

        # Compute Fisher Information Matrix using the test set
        for start_idx in range(0, x_test.shape[0], mini_batch_size):
            test_batch_x = x_test[start_idx:start_idx + mini_batch_size]
            test_batch_y = y_test[start_idx:start_idx + mini_batch_size]

            # Ensure batch tensors are on the correct device
            test_batch_x = test_batch_x.to(dev)
            test_batch_y = test_batch_y.to(dev)

            self.net.zero_grad()
            output, _ = self.net.predict(x=test_batch_x)
            loss = self.loss_func(output, test_batch_y.long())
            loss.backward()

            # Accumulate Fisher information
            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    fisher_matrix[name] += (param.grad ** 2)

        for name in fisher_matrix:
            fisher_matrix[name] /= len(range(0, x_test.shape[0], mini_batch_size))

        # Save the Fisher matrix and corresponding parameters
        self.fisher_matrices.append(fisher_matrix)
        self.prev_params_list.append({name: param.clone().detach() for name, param in self.net.named_parameters()})

    def update_ewc_loss(self):
        """Update the EWC loss by accumulating penalties for all tasks."""
        for task_idx, fisher_matrix in enumerate(self.fisher_matrices):
            prev_params = self.prev_params_list[task_idx]
            for name, param in self.net.named_parameters():
                #print(name) ändern um nur bestimmt layer zu locken
                if param.requires_grad and name in fisher_matrix:
                    fisher = fisher_matrix[name]
                    prev_param = prev_params[name]
                    self.ewc_penalty += (fisher * (param - prev_param) ** 2).sum()

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent with optional EWC regularization.
        :param x     : input
        :param target: desired output
        :return      : loss
        """
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target.long()) +self.lambda_ewc * self.ewc_penalty

        loss.backward(retain_graph=True)

        self.opt.step()
        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()

