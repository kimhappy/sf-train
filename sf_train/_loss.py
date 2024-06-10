import torch
import torch.nn as nn

_EPSILON = 0.00001

class _ESRLoss(nn.Module):
    def __init__(self):
        super(_ESRLoss, self).__init__()

    def forward(self, output, target):
        loss   = torch.add(target, -output)
        loss   = torch.pow(loss, 2)
        loss   = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + _EPSILON
        loss   = torch.div(loss, energy)
        return loss

class _DCLoss(nn.Module):
    def __init__(self):
        super(_DCLoss, self).__init__()

    def forward(self, output, target):
        loss   = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss   = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + _EPSILON
        loss   = torch.div(loss, energy)
        return loss

class _MixLoss(nn.Module):
    def __init__(self, lws):
        super(_MixLoss, self).__init__()
        self.lws = lws

    def forward(self, output, target):
        ret = 0.0

        for loss, weight in self.lws:
            ret += weight * loss(output, target)

        if ret < 0.0:
            print('Negative loss detected')

            for loss, weight in self.lws:
                print(f'loss: {loss(output, target):.6f} | weight: {weight:.6f}')

        return ret
