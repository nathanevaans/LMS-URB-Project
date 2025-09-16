import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class fbsde:
    def __init__(self, x_0, b, sigma, f, g, T, dim_x, dim_y, dim_d):
        self.x_0 = x_0.to(device)
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d


class Model(nn.Module):
    def __init__(self, equation, dim_h, method):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, equation.dim_y * equation.dim_d)
        self.y_0 = nn.Parameter(torch.rand(equation.dim_y, device=device))

        self.equation = equation
        self.method = method

    def forward(self, batch_size, N):
        def phi(_x):
            _x = F.relu(self.linear1(_x))
            _x = F.relu(self.linear2(_x))
            return self.linear3(_x).reshape(-1, self.equation.dim_y, self.equation.dim_d)

        delta_t = self.equation.T / N

        W = self.method(batch_size, self.equation.dim_d, N, delta_t)

        x = self.equation.x_0 + torch.zeros(W.size()[0], self.equation.dim_x, device=device)
        # y = self.y_0 + torch.zeros(W.size()[0], self.equation.dim_y, device=device)
        y = self.y_0.expand(W.size()[0], self.equation.dim_y)  # [batch_size, dim_y]

        for i in range(N):
            u = torch.cat((x, torch.ones(x.size()[0], 1, device=device) * delta_t * i), 1)
            z = phi(u)
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1)
            x = x + self.equation.b(delta_t * i, x, y) * delta_t + torch.matmul(self.equation.sigma(delta_t * i, x),
                                                                                w).reshape(-1, self.equation.dim_x)
            y = y - self.equation.f(delta_t * i, x, y, z) * delta_t + torch.matmul(z, w).reshape(-1,
                                                                                                 self.equation.dim_y)
        return x, y


class BSDEsolver:
    def __init__(self, equation, dim_h, method):
        self.model = Model(equation, dim_h, method).to(device)
        self.equation = equation

        self.best_loss = float("inf")
        self.best_y0 = None

    def train(self, batch_size, N, itr, run, lr=1e-3, folder=None, file_path=None):
        criterion = torch.nn.MSELoss().to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        loss_data, y0_data = [], []

        pbar = trange(itr, desc=f"{file_path}_{run}", unit="iter")
        for _ in pbar:
            x, y = self.model(batch_size, N)
            # loss = criterion(self.equation.g(x), y)
            loss = criterion(y.squeeze(), self.equation.g(x).squeeze())  # Both [batch_size]

            loss_data.append(float(loss))
            y0_data.append(float(self.model.y_0))

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.best_y0 = self.model.y_0.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(
                loss=f"{float(loss):.4f}",
                y0=f"{self.model.y_0.item():.4f}",
                best_y0=f"{self.best_y0:.4f}"
            )

        if file_path:
            np.save(f'{folder}/{file_path}_{run}_loss_data', loss_data)
            np.save(f'{folder}/{file_path}_{run}_y0_data', y0_data)
