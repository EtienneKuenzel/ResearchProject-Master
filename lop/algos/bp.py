import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.init as init
from torch import optim
from lop.algos.convGnT import ConvGnT
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_dim=128, num_classes=2):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.label_embed_dim = 16  # Embedding dimension for label

        # Embedding layer for labels
        self.label_emb = nn.Embedding(num_classes, self.label_embed_dim)

        # Encoder: image + label → latent (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + 1, 32, 4, 2, 1),  # input = [B, 4, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),     # [B, 64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),    # [B, 128, 8, 8]
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, z_dim)

        # Decoder: z + label → image
        self.decoder_input = nn.Linear(z_dim + self.label_embed_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # [B, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # [B, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # [B, 3, 32, 32]
            nn.Sigmoid()  # normalize output to [0,1]
        )

    def sample(self, n_samples, device='cuda'):
        half = n_samples // 2

        # Create labels: half 0s, half 1s
        y = torch.cat([
            torch.zeros(half, dtype=torch.long),
            torch.ones(n_samples - half, dtype=torch.long)
        ]).to(device)

        # Sample z from standard normal
        z = torch.randn(n_samples, self.z_dim).to(device)
        gen_x = self.decode(z, y)  # shape [n_samples, 3, 32, 32]
        gen_y = y  # shape [n_samples]

        return gen_x, gen_y
    def encode(self, x, y):
        y_img = self.label_emb(y).unsqueeze(2).unsqueeze(3)  # [B, embed, 1, 1]
        y_img = y_img.expand(-1, -1, x.size(2), x.size(3))   # [B, embed, 32, 32]
        y_img = y_img[:, :1, :, :]  # use 1 channel for label to match image
        x_input = torch.cat([x, y_img], dim=1)  # [B, 4, 32, 32]
        h = self.encoder(x_input)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.label_emb(y)
        z_cond = torch.cat([z, y_emb], dim=1)
        h = self.decoder_input(z_cond)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar


class ConvCBP(object):
    """
    The Continual Backprop algorithm
    """
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta=0.9, beta_2=0.999, replacement_rate=0.0001,
                 decay_rate=0.9, init='kaiming', util_type='contribution', maturity_threshold=100, device='cpu',
                 momentum=0, weight_decay=0):
        self.net = net
        self.opt = optim.SGD(self.net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        self.loss_func = F.cross_entropy
        self.previous_features = None
        self.gnt = ConvGnT(
            net=self.net.layers,
            hidden_activation=self.net.act_type,
            opt=optim.SGD(self.net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay),
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            init=init,
            num_last_filter_outputs=net.last_filter_output,
            util_type=util_type,
            maturity_threshold=maturity_threshold,
            device=device,
        )

    def learn(self, x, target):
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.gnt.gen_and_test(features=self.previous_features)
        return loss.detach(), output
class Backprop(object):
    def __init__(self, net, step_size=0.001, weight_decay=0.0, device='cpu', momentum=0):
        self.net = net
        self.device = device
        self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        self.loss_func = F.cross_entropy

    def prune_merge_neurons(self, task_activations, task_idx):
        task_activations = task_activations.cpu()
        for layer_idx, layer_offset in enumerate([-5, -3]):
            nlist = []
            target_layer_offset = layer_offset + 2
            max_weight_old = torch.max(self.net.layers[target_layer_offset].weight).item()
            bias_mean = torch.mean(self.net.layers[layer_offset].bias.data)
            bias_std = torch.std(self.net.layers[layer_offset].bias.data)
            weight_mean = torch.mean(self.net.layers[layer_offset].weight.data)
            weight_std = torch.std(self.net.layers[layer_offset].weight.data)

            weight_std1 = torch.std(self.net.layers[target_layer_offset].weight.data)
            weight_mean1 = torch.mean(self.net.layers[target_layer_offset].weight.data)

            for x in range(len(self.net.layers[layer_offset].weight.data)):
                activation_x = task_activations[task_idx, 0, layer_idx, x].flatten().cpu().numpy()
                if np.std(np.maximum(0, activation_x)) == 0:
                    init.normal_(self.net.layers[layer_offset].weight.data[x], mean=weight_mean, std=weight_std)
                    continue

                for y in range(x + 1, len(self.net.layers[layer_offset].weight.data)):
                    if x in nlist or y in nlist:
                        continue

                    data_x = np.maximum(0, task_activations[task_idx, 0, layer_idx, x].flatten().cpu().numpy())
                    data_y = np.maximum(0, task_activations[task_idx, 0, layer_idx, y].flatten().cpu().numpy())

                    if np.std(data_x) == 0 or np.std(data_y) == 0:
                        continue

                    correlation = np.corrcoef(data_x, data_y)[0, 1]
                    if correlation > 0.95:
                        for neuron in range(len(self.net.layers[target_layer_offset].weight.data)):
                            adjustment = self.net.layers[target_layer_offset].weight.data[neuron][y] * (np.std(data_x) / np.std(data_y))
                            self.net.layers[target_layer_offset].weight.data[neuron][x] += adjustment
                            init.normal_(self.net.layers[target_layer_offset].weight.data[neuron][y], mean=weight_mean1,std=weight_std1)
                        # Reset values of the merged (consumed) neuron
                        init.normal_(self.net.layers[layer_offset].bias.data[y], mean=bias_mean, std=bias_std)
                        init.normal_(self.net.layers[layer_offset].weight.data[y], mean=weight_mean, std=weight_std)

                        nlist.extend([y, x])

            max_weight_new = torch.max(self.net.layers[target_layer_offset].weight).item()
            self.net.layers[target_layer_offset].weight.data *= (max_weight_old / max_weight_new)
    def learn(self, x, target, task,decrease=0 ):
        layer_scaling = {
            "conv1.weight": 1-(decrease*5),
            "conv1.bias": 1-(decrease*5),
            "conv2.weight": 1-(decrease*4),
            "conv2.bias": 1-(decrease*4),
            "conv3.weight": 1-(decrease*3),
            "conv3.bias": 1-(decrease*3),
            "fc1.weight": 1-(decrease*2),
            "fc1.bias": 1-(decrease*2),
            "fc2.weight": 1-(decrease*1),
            "fc2.bias": 1-(decrease*1),
            "fc3.weight": 1.0,
            "fc3.bias": 1.0}
        layer_scaling = {name: scale + (1 - scale) * 1.005 ** -task for name, scale in layer_scaling.items()}
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target.long())

        loss.backward()
        for name, param in self.net.named_parameters():
            if name in layer_scaling:
                param.grad *= layer_scaling[name]
        self.opt.step()
        return loss.detach(), output.detach()
class EWC_Policy(object):
    def __init__(self, net, step_size=0.001, loss='nll', weight_decay=0.0,opt="s",to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0, lambda_ewc=1):
        self.net = net.to(device)
        self.device = device
        self.opt = optim.Adam(self.net.parameters(), lr=step_size, weight_decay=weight_decay)
        self.loss = loss
        self.crit = nn.CrossEntropyLoss()
        self.loss_func = nn.CrossEntropyLoss()
        self.weight =100000000000000
    def _update_mean_params(self):
        for param_name, param in self.net.named_parameters():
            _buff_param_name = param_name.replace('.', '__')
            self.net.register_buffer(_buff_param_name+'_estimated_mean', param.data.clone())

    def _update_fisher_params(self, x_train, y_train, batch_size, num_batch):
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)  # adapt dtype if necessary
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.long)

        # Wrap data into a Dataset and DataLoader
        train_dataset = TensorDataset(x_train, y_train)
        dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        log_liklihoods = []
        for i, (input, target) in enumerate(dl):
            if i > num_batch:
                break
            output = F.log_softmax(self.net.predict(input.to(self.device))[0], dim=1)
            log_liklihoods.append(output[:, target])
        log_likelihood = torch.cat(log_liklihoods).mean()
        grad_log_liklihood = autograd.grad(log_likelihood, self.net.parameters())
        _buff_param_names = [param[0].replace('.', '__') for param in self.net.named_parameters()]
        for _buff_param_name, param in zip(_buff_param_names, grad_log_liklihood):
            self.net.register_buffer(_buff_param_name + '_estimated_fisher', param.data.clone() ** 2)

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
