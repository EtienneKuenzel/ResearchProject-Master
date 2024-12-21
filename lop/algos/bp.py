import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from copy import deepcopy
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

class DQN_EWC_Policy(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0, lambda_ewc=0.5):
        self.net = net
        self.device = device
        self.lambda_ewc = lambda_ewc  # Regularization strength for EWC

        # Define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                  weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)

        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder for Fisher matrix and old parameters
        self.fisher_matrix = None
        self.params_old = None
        self.ewc_penalty = 0.0

    def update_ewc_penalty(self, dataset,dev):
        fisher = {}
        for n, p in deepcopy({n: p for n, p in self.net.named_parameters() if p.requires_grad}).items():
            fisher[n] = torch.zeros_like(p)
        _, _, x_test, y_test = dataset
        x_test = x_test.float()
        x_test, y_test = x_test.to(dev), y_test.to(dev)
        mini_batch_size=100
        self.net.eval()
        for i, start_idx in enumerate(range(0, x_test.shape[0], mini_batch_size)):
            test_batch_x = x_test[start_idx:start_idx + mini_batch_size]
            test_batch_y = y_test[start_idx:start_idx + mini_batch_size]

            self.net.zero_grad()
            output, _ = self.net.predict(test_batch_x)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), test_batch_y.long())
            negloglikelihood.backward()

            for n, p in self.net.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) / len(test_batch_x)
        self.fisher_matrix = fisher
        self.params_old = {n: p.clone().detach() for n, p in self.net.named_parameters() if p.requires_grad}

        loss = 0.0
        for n, p in self.net.named_parameters():
            if n in self.fisher_matrix:
                loss += (self.fisher_matrix[n] * (p - self.params_old[n]) ** 2).sum()

        self.ewc_penalty += loss
    def learn(self, x, target):
        """
        Learn using one step of gradient descent with optional EWC regularization.
        :param x: input
        :param target: desired output
        :return: loss
        """
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target.long()) + self.lambda_ewc * self.ewc_penalty

        loss.backward(retain_graph=True)

        self.opt.step()
        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()


