import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional, Any
import torch.utils.checkpoint as checkpoint
class ConvNet(nn.Module):
    def __init__(self, num_classes=2, activation="relu"):
        """
        Flexible Convolutional Neural Network with configurable activation functions.
        Supports both ReLU and PAU activations.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.last_filter_output = 2 * 2
        self.num_conv_outputs = 128 * self.last_filter_output
        self.fc1 = nn.Linear(self.num_conv_outputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

        # Determine activation function
        if activation.lower() == "relu":
            self.act_fn = nn.ReLU
        elif activation.lower() == "sig":
            self.act_fn = nn.Sigmoid
        elif activation.lower() == "tanh":
            self.act_fn = nn.Tanh
        elif activation.lower() == "leakrelu":
            self.act_fn = nn.LeakyReLU
        # Architecture
        self.layers = nn.ModuleList([
            self.conv1, self.act_fn(),
            self.conv2, self.act_fn(),
            self.conv3, self.act_fn(),
            self.fc1, self.act_fn(),
            self.fc2, self.act_fn(),
            self.fc3
        ])

        self.act_type = activation.lower()

    def predict(self, x):
        x1 = self.pool(self.layers[1](self.layers[0](x)))
        x2 = self.pool(self.layers[3](self.layers[2](x1)))
        x3 = self.pool(self.layers[5](self.layers[4](x2)))
        x3 = x3.view(-1, self.num_conv_outputs)
        x4 = self.layers[7](self.layers[6](x3))
        x5 = self.layers[9](self.layers[8](x4))
        x6 = self.layers[10](x5)
        return x6, [x1, x2, x3, x4, x5]


class ConvNet_PAU(nn.Module):
    def __init__(self, num_classes=2):
        """
        Convolutional Neural Network with 3 convolutional layers followed by 3 fully connected layers
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.last_filter_output = 2 * 2
        self.num_conv_outputs = 128 * self.last_filter_output
        self.fc1 = nn.Linear(self.num_conv_outputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

        # architecture
        self.layers = nn.ModuleList()
        self.layers.append(self.conv1) #0
        self.layers.append(nn.ReLU())#1
        self.layers.append(self.conv2)#2
        self.layers.append(nn.ReLU())#3
        self.layers.append(self.conv3)#4
        self.layers.append(nn.ReLU())#5
        self.layers.append(self.fc1)#6
        self.layers.append(PAU(cuda=torch.cuda.is_available()).requires_grad_(True))#7
        self.layers.append(self.fc2)#8
        self.layers.append(PAU(cuda=torch.cuda.is_available()).requires_grad_(True))#9
        self.layers.append(self.fc3)#10

        self.act_type = 'relu'

    def predict(self, x):
        x1 = self.pool(self.layers[1](self.layers[0](x)))
        x2 = self.pool(self.layers[3](self.layers[2](x1)))
        x3 = self.pool(self.layers[5](self.layers[4](x2)))
        x3 = x3.view(-1, self.num_conv_outputs)
        x4 = self.layers[7](self.layers[6](x3))
        x5 = self.layers[9](self.layers[8](x4))
        x6 = self.layers[10](x5)
        return x6, [x1, x2, x3, x4, x5]
class ConvNet_TENT(nn.Module):
    def __init__(self, num_classes=2):
        """
        Convolutional Neural Network with 3 convolutional layers followed by 3 fully connected layers
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.last_filter_output = 2 * 2
        self.num_conv_outputs = 128 * self.last_filter_output
        self.fc1 = nn.Linear(self.num_conv_outputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

        # architecture
        self.layers = nn.ModuleList()
        self.layers.append(self.conv1) #0
        self.layers.append(nn.ReLU())#1
        self.layers.append(self.conv2)#2
        self.layers.append(nn.ReLU())#3
        self.layers.append(self.conv3)#4
        self.layers.append(nn.ReLU())#5
        self.layers.append(self.fc1)#6
        self.layers.append(Tent())#7
        self.layers.append(self.fc2)#8
        self.layers.append(Tent())#9
        self.layers.append(self.fc3)#10

        self.act_type = 'relu'

    def predict(self, x):
        x1 = self.pool(self.layers[1](self.layers[0](x)))
        x2 = self.pool(self.layers[3](self.layers[2](x1)))
        x3 = self.pool(self.layers[5](self.layers[4](x2)))
        x3 = x3.view(-1, self.num_conv_outputs)
        x4 = self.layers[7](self.layers[6](x3))
        x5 = self.layers[9](self.layers[8](x4))
        x6 = self.layers[10](x5)
        return x6, [x1, x2, x3, x4, x5]

class PAU(nn.Module):
    """
    This class implements the Pade Activation Unit proposed in:
    https://arxiv.org/pdf/1907.06732.pdf
    """

    def __init__(
            self,
            weights = [torch.tensor((0.02996348, 0.61690165, 2.37539147, 3.06608078, 1.52474449, 0.25281987),dtype=torch.float), torch.tensor((1.19160814, 4.40811795, 0.91111034, 0.34885983),dtype=torch.float)],
            m: int = 5,
            n: int = 4,
            efficient: bool = True,
            eps: float = 1e-08,
            activation_unit = 5 ,
            **kwargs: Any
    ) -> None:
        """
        Constructor method
        :param m (int): Size of nominator polynomial. Default 5.
        :param n (int): Size of denominator polynomial. Default 4.
        :param initial_shape (Optional[str]): Initial shape of PAU, if None random shape is used, also if m and n are
        not the default value (5 and 4) a random shape is utilized. Default "leaky_relu_0_2".
        :param efficient (bool): If true efficient variant with checkpointing is used. Default True.
        :param eps (float): Constant for numerical stability. Default 1e-08.
        :param **kwargs (Any): Unused
        """
        # Call super constructor
        super(PAU, self).__init__()
        # Save parameters
        self.efficient: bool = efficient
        self.m: int = m
        self.n: int = n
        self.eps: float = eps
        self.initial_weights = weights
        # Init weights
        weights_nominator, weights_denominator = self.initial_weights
        self.weights_nominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(weights_nominator.view(1, -1))
        self.weights_denominator: Union[nn.Parameter, torch.Tensor] = nn.Parameter(weights_denominator.view(1, -1))

    def freeze(self) -> None:
        """
        Function freezes the PAU weights by converting them to fixed model parameters.
        """

        if isinstance(self.weights_nominator, nn.Parameter):
            weights_nominator = self.weights_nominator.data.clone()
            del self.weights_nominator
            self.register_buffer("weights_nominator", weights_nominator)
        if isinstance(self.weights_denominator, nn.Parameter):
            weights_denominator = self.weights_denominator.data.clone()
            del self.weights_denominator
            self.register_buffer("weights_denominator", weights_denominator)
    def _forward(self,input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input (torch.Tensor): Input tensor of the shape [*]
        :return (torch.Tensor): Output tensor of the shape [*]
        """
        device = input.device
        self.weights_nominator = nn.Parameter(self.weights_nominator.to(device))
        self.weights_denominator = nn.Parameter(self.weights_denominator.to(device))
        # Save original shape
        shape: Tuple[int, ...] = input.shape
        # Flatten input tensor
        input: torch.Tensor = input.view(-1)
        if self.efficient:
            # Init nominator and denominator
            nominator: torch.Tensor = torch.ones_like(input=input) * self.weights_nominator[..., 0]
            denominator: torch.Tensor = torch.zeros_like(input=input)
            # Compute nominator and denominator iteratively
            for index in range(1, self.m + 1):
                x: torch.Tensor = (input ** index)
                nominator: torch.Tensor = nominator + x * self.weights_nominator[..., index]
                if index < (self.n + 1):
                    denominator: torch.Tensor = denominator + x * self.weights_denominator[..., index - 1]
            denominator: torch.Tensor = denominator + 1.
        else:
            # Get Vandermonde matrix
            vander_matrix: torch.Tensor = torch.vander(x=input, N=self.m + 1, increasing=True)
            # Compute nominator
            nominator: torch.Tensor = (vander_matrix * self.weights_nominator).sum(-1)
            # Compute denominator
            denominator: torch.Tensor = 1. + torch.abs((vander_matrix[:, 1:self.n + 1]
                                                        * self.weights_denominator).sum(-1))
        # Compute output and reshape
        output: torch.Tensor = (nominator / denominator.clamp(min=self.eps)).view(shape)
        return output

    def forward(self,input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input (torch.Tensor): Input tensor of the shape [batch size, *]
        :return (torch.Tensor): Output tensor of the shape [batch size, *]
        """
        # Make input contiguous if needed
        input: torch.Tensor = input if input.is_contiguous() else input.contiguous()
        if self.efficient:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input=input)

class Tent(nn.Module):
    def __init__(self):
         super(Tent, self).__init__()

    def forward(self, input):
        out = nn.functional.relu(input + 1) - 2 * nn.functional.relu(input) + nn.functional.relu(input - 1)
        return out