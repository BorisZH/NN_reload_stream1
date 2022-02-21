# velocity = momentum (0.9-0.99)* velocity - lr*gradient
# w = w + velocity (скорость-вектор)

import torch


class SDGMomentum:
    def __init__(self, momentum, lr, model):
        self.momentum = momentum
        self.lr = lr
        self.velocity = torch.zeros_like(model)
        self.model = model

    def step(self, grad):
        self.velocity = self.momentum * self.velocity - self.lr * grad
        self.model += self.velocity


# найти пересечение с осью x линии y = 2 * x - 7 методом градиентного спуска

def func(x):
    return (2 * x -7) ** 2

def grad_f(x):
    return 2 * (2 * x - 7)

def solver(init_x):
    x = torch.tensor(init_x)
    g = grad_f(x)
    optim = SDGMomentum(0.9, 0.001, x)
    for i in range(1000):
        optim.step(g)
        g = grad_f(optim.model)
    print(optim.model)

solver(6.)
