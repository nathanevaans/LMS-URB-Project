import random

import torch
import numpy as np

from solver import BSDEsolver
from solver import fbsde


def set_seed(_seed):
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def brownian(_batch_size, _dim_d, _N, _delta_t):
    return torch.randn(_batch_size, _dim_d, _N, device=device) * np.sqrt(_delta_t)


def step_size(_batch_size, _dim_d, _N, _delta_t):
    return (2 * torch.randint(0, 2, (_batch_size, _dim_d, _N), device=device) - 1).float() * np.sqrt(_delta_t)


dim_x, dim_y, dim_d, dim_h, N, itr, batch_size = 10, 1, 1, 20, 100, 3000, 1000

x_0, T = torch.ones(dim_x), 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

A = torch.rand(dim_x, device=device)
B = torch.rand(dim_x, device=device)
S = torch.rand(dim_x, device=device)


def b(t, x, y):
    return (A * (B - x)).reshape(batch_size, dim_x)


def sigma(t, x):
    return (S * torch.sqrt(torch.abs(x))).reshape(batch_size, dim_x, dim_d)


def f(t, x, y, z):
    return -y * torch.max(x, 1)[0].reshape(batch_size, dim_y)


def g(x):
    return torch.ones(batch_size, dim_y, device=device)


equation = fbsde(x_0, b, sigma, f, g, T, dim_x, dim_y, dim_d)

seeds = [random.randint(0, 2**32 - 1) for _ in range(4)]
offset = 1
for idx, seed in enumerate(seeds):
    set_seed(seed)
    bsde_solver = BSDEsolver(equation, dim_h, brownian)
    bsde_solver.train(batch_size, N, itr, idx + offset, folder="multi_cir_data", file_path="Brownian")

    set_seed(seed)
    bsde_solver = BSDEsolver(equation, dim_h, step_size)
    bsde_solver.train(batch_size, N, itr, idx + offset, folder="multi_cir_data", file_path="Step_size")
