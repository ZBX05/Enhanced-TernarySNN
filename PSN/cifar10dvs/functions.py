import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# def TET_loss(outputs, labels, criterion, means, lamb):
#     T = outputs.size(1)
#     Loss_es = 0
#     for t in range(T):
#         Loss_es += criterion(outputs[:, t, ...], labels)
#     Loss_es = Loss_es / T # L_TET
#     if lamb != 0:
#         MMDLoss = torch.nn.MSELoss()
#         y = torch.zeros_like(outputs).fill_(means)
#         Loss_mmd = MMDLoss(outputs, y) # L_mse
#     else:
#         Loss_mmd = 0
#     return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total

def TET_loss(outputs, labels, criterion, means, lamb, mem, lamb_mem=0.01):
    T = outputs.size(1)
    Loss_es = 0
    mem_loss = 0
    N = len(mem)
    MMDLoss = torch.nn.MSELoss()
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
        for n in range(N):
            mem_loss += (lamb_mem / (t + 1)) * MMDLoss(mem[n][t], torch.zeros_like(mem[n][t], device=mem[n][t].device)).to(Loss_es.device)
    Loss_es = Loss_es / T # L_TET
    mem_loss = mem_loss / (N * T)
    if lamb != 0:
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd + mem_loss # L_Total