import torch
from torch._C import device
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def nll_loss(output, target):
    return F.nll_loss(output, target)


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze())
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
    
