import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class Identity(nn.Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return x


class FFTExpandBigConv1d(nn.Module):
    # 专为小输入大卷积核设计
    # 输入：N, 1, l_in
    # 输出：N, num_experts, l_out
    def __init__(
        self,
        num_experts: int,
        input_len: int,
        output_len: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.input_len = input_len
        self.output_len = output_len

        # we pad x 1% of length on each end... for bizzare reasons..
        self.pad_len = math.floor((self.input_len + self.output_len - 1) / 100)
        self.pad_len = max(self.pad_len, 1)

        self.time_len = self.input_len + self.output_len - 1 + 2 * self.pad_len
        self.freq_len = self.time_len // 2 + 1

        # Initialized as Average filter
        self.weight = nn.Parameter(
            torch.zeros((1, self.num_experts, 1, self.freq_len),
                        dtype=torch.cfloat))
        self.weight.data[..., 0] = 1
        self.bias = nn.Parameter(
            torch.zeros(1,
                        self.num_experts,
                        1,
                        self.freq_len,
                        dtype=torch.cfloat))

    def forward(self, x: torch.Tensor, rank_experts=None):
        if rank_experts is not None:
            weight = self.weight * rank_experts  # 1, num_experts, input_len//2+1
            weight = weight.sum(dim=1, keepdim=True)  # 1, 1, input_len//2+1

            bias = self.bias * rank_experts
            bias = bias.sum(dim=1, keepdim=True)
        else:
            weight = self.weight
            bias = self.bias

        # input: N, 1, l_in
        # depth-wise convolution
        # pad x to match kernel

        x = F.pad(x, [self.pad_len, self.output_len - 1 + self.pad_len])

        # calculate FFT
        x = torch.fft.rfft(x)
        # weight = torch.fft.rfft(weight)

        # frequency production
        x = x * weight

        # bias
        x = x + bias

        # invert FFT
        x = torch.fft.irfft(x, n=self.time_len)

        x = x[..., -self.output_len - self.pad_len:-self.pad_len]

        # output: N, experts, l_out if rank is None else N, 1, l_out
        return x


class StaticTimeWeight(nn.Module):

    def __init__(self, input_len, num_experts):
        super().__init__()
        self.input_len = input_len
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.ones(1, self.num_experts, 1, self.input_len))

    def forward(self, x, rank_experts=None):
        # x: N, 1, c, input_len
        # if rank_experts provided, output is N, 1, c, input_len//2+1
        # if not, output is N, num_experts, c, input_len//2+1

        if rank_experts is not None:
            weight = self.weight * rank_experts  # 1, num_experts, c, input_len//2+1
            weight = weight.sum(dim=1, keepdim=True)  # 1, 1, c, input_len//2+1
        else:
            weight = self.weight
        x = x * weight
        return x


class StaticFreqWeight(nn.Module):
    # we do not use window function since it is a linear operation

    def __init__(self, input_len, num_experts):
        super().__init__()
        self.input_len = input_len
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.ones(1, self.num_experts, 1, self.input_len // 2 + 1))

    def get_weight_channel(self, rank_experts):

        if rank_experts is not None:
            weight = self.weight * rank_experts  # 1, num_experts, c, input_len//2+1
            weight = weight.sum(dim=1, keepdim=True)  # 1, 1, c, input_len//2+1
        else:
            weight = self.weight
        return weight

    def forward(self, x, rank_experts=None, windowing=False):
        # x: N, 1, c, input_len
        # if rank_experts provided, output is N, 1, c, input_len//2+1
        # if not, output is N, num_experts, c, input_len//2+1

        weight = self.get_weight_channel(rank_experts)

        # x = F.pad(x, [self.input_len // 2, self.input_len // 2])
        if windowing:
            window = torch.hamming_window(self.input_len,
                                          dtype=x.dtype,
                                          device=x.device)
            x = x * window
        x = torch.fft.rfft(x)
        x = x * weight
        x = torch.fft.irfft(x, n=self.input_len)
        if windowing:
            x = x / window
        # x = x[:, :, :, self.input_len // 2:-self.input_len // 2]

        return x


class DiPE(nn.Module):

    def __init__(
        self,
        input_len: int,
        output_len: int,
        input_features: int,
        output_features: int,
        individual_f: bool = False,
        individual_t: bool = False,
        individual_c: bool = False,
        num_experts: int = 1,
        use_revin: bool = True,
        use_time_w: bool = True,
        use_freq_w: bool = True,
        loss_alpha: float = 0.,
        t_loss: Literal['mse', 'mae'] = 'mse',
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_features = input_features
        self.individual_f = individual_f
        self.individual_t = individual_t
        self.individual_c = individual_c
        self.num_experts = num_experts
        assert input_features == output_features

        self.use_revin = use_revin
        self.use_time_w = use_time_w
        self.use_freq_w = use_freq_w
        self.loss_alpha = loss_alpha
        self.t_loss = t_loss

        self.example_input_array = torch.Tensor(32, input_len, input_features)

        if self.num_experts > 1:
            self.route = nn.Parameter(
                torch.randn(1, num_experts, self.num_features, 1))
            self.temperature = 114514
            self.temperature = float('nan')
            self.router_softmax = nn.Softmax(dim=1)
        # self.static_route = torch.eye(self.num_experts).unsqueeze(0).unsqueeze(-1)
        self.static_route = torch.eye(
            self.num_features).unsqueeze(0).unsqueeze(-1)

        if self.use_time_w:
            if self.individual_t:
                self.time_w = StaticTimeWeight(self.input_len,
                                               self.num_features)
            else:
                self.time_w = StaticTimeWeight(self.input_len, self.num_experts)
        else:
            self.time_w = Identity()

        if self.use_freq_w:
            if self.individual_f:
                self.freq_w = StaticFreqWeight(self.input_len,
                                               self.num_features)
            else:
                self.freq_w = StaticFreqWeight(self.input_len, self.num_experts)
        else:
            self.freq_w = Identity()

        if self.individual_c:
            self.expert = FFTExpandBigConv1d(self.num_features, self.input_len,
                                             self.output_len)
        else:
            self.expert = FFTExpandBigConv1d(self.num_experts, self.input_len,
                                             self.output_len)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.shape[0]

        x = rearrange(x, 'n l c -> n 1 c l')

        if self.use_revin:

            x_mean = x.mean(dim=-1, keepdim=True).detach()
            x_std = x.std(dim=-1, keepdim=True).detach().clamp(min=1e-7)
            x = (x - x_mean) / x_std

        if self.num_experts > 1:

            rank_experts = self.router_softmax(self.route /
                                               self.temperature)  # 1, h, c, 1

        else:
            rank_experts = None

        if self.individual_f:
            x = self.freq_w(x, self.static_route.to(x.device))
        else:
            x = self.freq_w(x, rank_experts)
        x = self.dropout(x)

        if self.individual_t:
            x = self.time_w(x, self.static_route.to(x.device))
        else:
            x = self.time_w(x, rank_experts)

        if self.individual_c:
            x = self.expert(x, self.static_route.to(x.device))
        else:
            x = self.expert(x, rank_experts)

        if self.use_revin:
            x = x * x_std
            x = x + x_mean

        x = rearrange(x, 'n 1 c l -> n l c')

        return x

    def loss(self, y, y_hat):
        y = rearrange(y, 'n l c -> n c l')
        y_hat = rearrange(y_hat, 'n l c -> n c l')

        if self.t_loss == 'mse':
            time_loss = F.mse_loss(y, y_hat)
        else:
            time_loss = F.l1_loss(y, y_hat)

        if self.use_freq_w:

            if self.num_experts > 1:
                rank_experts = self.router_softmax(
                    self.route / self.temperature)  # 1, h, c, 1
            else:
                rank_experts = None

            if self.individual_f:
                rank_experts = self.static_route.to(y.device)

            freq_w = self.freq_w.get_weight_channel(rank_experts)
            freq_w = freq_w.detach()
            freq_w = freq_w / freq_w.mean(dim=-1, keepdim=True)

            if freq_w.shape[-1] != y.shape[-1] // 2 + 1:
                with torch.no_grad():
                    freq_w = torch.fft.irfft(freq_w, n=y.shape[-1])
                    freq_w = torch.fft.rfft(freq_w)

        else:
            freq_w = 1

        fft_y = torch.fft.rfft(y, norm='ortho')
        fft_y_hat = torch.fft.rfft(y_hat, norm='ortho')

        freq_loss = F.l1_loss(fft_y * freq_w, fft_y_hat * freq_w)

        return (1 - self.loss_alpha) * time_loss + self.loss_alpha * freq_loss
