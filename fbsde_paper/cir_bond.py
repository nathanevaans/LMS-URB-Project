import random

import torch

import numpy as np

from solver import BSDEsolver
from solver import fbsde

dim_x, dim_y, dim_d, dim_h, N, itr, batch_size = 1, 1, 1, 11, 100, 3000, 1000

x_0, T = torch.ones(dim_x), 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def b(t, x, y):
    return (1 - x).reshape(batch_size, dim_x)


def sigma(t, x):
    return torch.sqrt(torch.abs(x)).reshape(batch_size, dim_x, dim_d)


def f(t, x, y, z):
    return (-y * x).reshape(batch_size, dim_y)


def g(x):
    return torch.ones(batch_size, dim_y, device=device)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def brownian(_batch_size, _dim_d, _N, _delta_t):
    return torch.randn(_batch_size, _dim_d, _N, device=device) * np.sqrt(_delta_t)


def step_size(_batch_size, _dim_d, _N, _delta_t):
    return (2 * torch.randint(0, 2, (_batch_size, _dim_d, _N), device=device) - 1).float() * np.sqrt(_delta_t)


equation = fbsde(x_0, b, sigma, f, g, T, dim_x, dim_y, dim_d)



seeds = [random.randint(0, 2**32 - 1) for _ in range(10)]
offset = 401
for idx, seed in enumerate(seeds):
    set_seed(seed)
    bsde_solver = BSDEsolver(equation, dim_h, brownian)
    bsde_solver.train(batch_size, N, itr, idx + offset, folder="cir_data", file_path="Brownian")

    set_seed(seed)
    bsde_solver = BSDEsolver(equation, dim_h, step_size)
    bsde_solver.train(batch_size, N, itr, idx + offset, folder="cir_data", file_path="Step_size")