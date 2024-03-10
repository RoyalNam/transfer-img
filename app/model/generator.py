import torch
from torch import nn


class ResiduaBlock(nn.Module):
  def __init__(self, input_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect'),
        nn.InstanceNorm2d(input_channels),
        nn.ReLU()
    )

  def forward(self, x):
    return self.conv(x) + x


class ContractingBlock(nn.Module):
  def __init__(self, input_channels, use_bn=True, kernel_size=3, act='relu'):
    super().__init__()
    self.conv1 = nn.Conv2d(
            input_channels,
            input_channels*2,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            padding_mode='reflect'
        )
    self.instanceNorm = nn.InstanceNorm2d(input_channels*2)
    self.act = nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2)
    self.use_bn = use_bn

  def forward(self, x):
    x = self.conv1(x)
    if self.use_bn:
      x = self.instanceNorm(x)
    x = self.act(x)
    return x


class ExplandingBlock(nn.Module):
  def __init__(self, input_channels, use_bn=True):
    super().__init__()
    self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, 3, 2, 1, output_padding=1)
    self.instanceNorm = nn.InstanceNorm2d(input_channels // 2)
    self.act = nn.ReLU()

    self.use_bn = use_bn

  def forward(self, x):
    x = self.conv1(x)
    if self.use_bn:
        x = self.instanceNorm(x)
    return self.act(x)


class FeatureMapBlock(nn.Module):
  def __init__(self, input_channels, output_channels):
    super().__init__()
    self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

  def forward(self, x):
    return self.conv(x)


class Generator(nn.Module):
  def __init__(self, input_channels, output_channels, hidden_channels=64):
    super().__init__()
    self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
    self.contract1 = ContractingBlock(hidden_channels)
    self.contract2 = ContractingBlock(hidden_channels*2)
    res_mult = 4
    self.res = nn.Sequential(
            *[ResiduaBlock(hidden_channels * res_mult) for _ in range(9)]
        )
    self.expand2 = ExplandingBlock(hidden_channels*4)
    self.expand3 = ExplandingBlock(hidden_channels*2)
    self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
    self.tanh = nn.Tanh()

  def forward(self, x):
    x = self.upfeature(x)
    x = self.contract1(x)
    x = self.contract2(x)
    x = self.res(x)
    x = self.expand2(x)
    x = self.expand3(x)
    x = self.downfeature(x)
    return self.tanh(x)