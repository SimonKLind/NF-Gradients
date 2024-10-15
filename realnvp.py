import torch
import torch.nn as nn
import torch.nn.functional as F

class STBlock(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.lin1 = nn.Linear(n_in, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.s = nn.Linear(n_hidden, n_out)
        self.t = nn.Linear(n_hidden, n_out)
        self.s_scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        x = F.relu(self.lin1(x), inplace=True)
        x = F.relu(self.lin2(x), inplace=True)
        s = self.s_scale * torch.tanh(self.s(x))
        t = self.t(x)
        return s, t
        
class CouplingLayer(nn.Module):
    def __init__(self, n1, n2, n_hidden):
        super().__init__()
        self.st1 = STBlock(n2, n_hidden, n1)
        self.st2 = STBlock(n1, n_hidden, n2)

    def forward(self, x1, x2, log_det):
        s, t = self.st1(x2)
        x1 = x1 * torch.exp(s) + t
        log_det += torch.sum(s, dim=1)

        s, t = self.st2(x1)
        x2 = x2 * torch.exp(s) + t
        log_det += torch.sum(s, dim=1)

        return x1, x2, log_det

class RealNVP(nn.Module):
    def __init__(self, input_dim, n_hidden, n_coupling):
        super().__init__()

        half = int(input_dim / 2)
        rest = int(input_dim - half)

        self.input_dim = input_dim
        self.half = half

        self.coupling_layers = nn.ModuleList([CouplingLayer(half, rest, n_hidden) for _ in range(n_coupling)])

    def forward(self, x):
        x1 = x[:, :self.half]
        x2 = x[:, self.half:]

        batch = x.size(0)
        log_det = torch.zeros(batch, device=x.device)

        for layer in self.coupling_layers:
            x1, x2, log_det = layer(x1, x2, log_det)

        x = torch.concat((x1, x2), dim=1)

        # Formula for multivariate gaussian log-likelihood with mean=0 covariance=1
        #                                                                              log(2*pi)
        log_Pu = -0.5 * torch.sum(torch.square(x), dim=1) - 0.5 * self.input_dim * 1.8378770664093453

        log_Px = log_Pu + log_det

        return log_Px, log_Pu, log_det
        

def main():
    nvp = RealNVP(input_dim=765, n_hidden=512, n_coupling=5)
    inp = torch.randn((8, 765))
    log_Px, log_Pu, log_det = nvp(inp)
    print(log_Px, log_Pu, log_det)

if __name__ == '__main__': main()
