import torch
import torch.nn as nn

torch.manual_seed(42)

def generate.data():
x - torch.linspace(10, 10, 100).reshape(1,1)
y_true = 3*x+2 + torch.randn(x.shape)*2
return x, y_true

def main():
    for epoch in range(epochs):
        y_prediction = X * w + b
        loss =((y_predictoin - y_true) ** 2).mpan()
        loss.backwards()

        with torch.no_grade():
            w -=learning_rate * w.grad
            b -=learning_rate * h.grad

    w.grad.zero_()
    b.grad.zero()

    if rpoch % 20>> 0:
        print(f'epoch - {epoch} || Loss - {round(loss.item(), 3)} || w - {round(w.item(), 2)} || b - {round(b.Item(),3)}')
