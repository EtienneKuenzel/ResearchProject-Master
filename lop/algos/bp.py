import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.autograd as autograd
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
        self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        self.loss = loss
        self.crit = nn.CrossEntropyLoss()
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        self.weight = 100
    def _update_mean_params(self):
        for param_name, param in self.net.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.net.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, x_train, y_train, batch_size, num_batch):
        # Make sure your data is in tensor form
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)  # adapt dtype if necessary
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)

        # Wrap data into a Dataset and DataLoader
        train_dataset = TensorDataset(x_train, y_train)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        log_likelihoods = []

        for i, (inputs, targets) in enumerate(dl):
            if i >= num_batch:
                break
            outputs = F.log_softmax(self.net.predict(inputs.to(self.device))[0], dim=1)
            # Select the log likelihood for the correct class per sample
            selected = outputs[range(outputs.shape[0]), targets]
            log_likelihoods.append(selected)

        log_likelihood = torch.cat(log_likelihoods).mean()

        grad_log_likelihood = autograd.grad(log_likelihood, self.net.parameters(), create_graph=False)

        _buff_param_names = [name.replace('.', '__') for name, _ in self.net.named_parameters()]
        for _buff_param_name, grad in zip(_buff_param_names, grad_log_likelihood):
            self.net.register_buffer(_buff_param_name + '_estimated_fisher', grad.data.clone() ** 2)

    def register_ewc_params(self, x,y, batch_size, num_batches):
        self._update_fisher_params(x,y, batch_size, num_batches)
        self._update_mean_params()

    def _compute_consolidation_loss(self, weight):
        try:
            losses = []
            for param_name, param in self.net.named_parameters():
                _buff_param_name = param_name.replace('.', '__')
                estimated_mean = getattr(self.net, '{}_estimated_mean'.format(_buff_param_name))
                estimated_fisher = getattr(self.net, '{}_estimated_fisher'.format(_buff_param_name))
                losses.append((estimated_fisher * (param - estimated_mean) ** 2).sum())
            return (weight / 2) * sum(losses)
        except AttributeError:
            return 0
    def learn(self, x, target):
        output, _ = self.net.predict(x.to(self.device))
        loss = self._compute_consolidation_loss(self.weight) + self.loss_func(output, target.long())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.detach(), output.detach()

