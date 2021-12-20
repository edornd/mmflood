import torch

from floods.utils.schedulers import PolynomialLRDecay


def test_polynomial_lr_decay():
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    scheduler = PolynomialLRDecay(optim, max_decay_steps=99, end_learning_rate=0.0001, power=3.0)

    for epoch in range(1, 20):
        scheduler.step(epoch)

        print(epoch, optim.param_groups[0]['lr'])
