import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from copy import deepcopy
class Backprop(object):
    def __init__(self, net, step_size=0.001, loss='nll', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
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

class EWC_Policy(object):
    def __init__(self, net, step_size=0.001, loss='nll', weight_decay=0.0,opt="s",to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0, lambda_ewc=1):
        self.net = net.to(device)
        self.device = device
        self.lambda_ewc = lambda_ewc  # Regularization strength for EWC
        self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder for Fisher matrix and old parameters
        self.fisher_matrix = None
        self.params_old = {n: p.clone().detach() for n, p in self.net.named_parameters() if p.requires_grad}
        self.ewc_penalty = 0.0

    def update_ewc_penalty(self, dataset):
        fisher = {}
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p, device=self.device)

        x_test, y_test,_,_ = dataset
        x_test, y_test = x_test.float().to(self.device), y_test.to(self.device)

        test_batch_x = x_test[0:1200]
        test_batch_y = y_test[0:1200]

        self.net.zero_grad()
        output, _ = self.net.predict(test_batch_x)
        negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), test_batch_y.long())
        negloglikelihood.backward(retain_graph=True)  # Ensuring graph is retained
        for n, p in self.net.named_parameters():
            if p.grad is not None:
                fisher[n] += (p.grad.data ** 2) / len(test_batch_x)  # Normalize per batch
        self.fisher_matrix = fisher

        # Compute initial EWC penalty
        loss = 0.0
        for n, p in self.net.named_parameters():
            if n in self.fisher_matrix:
                loss += (self.fisher_matrix[n] * (p - self.params_old[n]) ** 2).sum()
        self.ewc_penalty = loss
        self.params_old = {n: p.clone().detach() for n, p in self.net.named_parameters() if p.requires_grad}


    def learn(self, x, target):
        self.opt.zero_grad()
        output, _ = self.net.predict(x.to(self.device))
        loss = self.loss_func(output, target.long().to(self.device)) + self.lambda_ewc * self.ewc_penalty
        loss.backward(retain_graph=True)
        self.opt.step()
        return loss.detach(), output.detach()



