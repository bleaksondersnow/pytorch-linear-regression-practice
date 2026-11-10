
import torch
import torch.nn as nn

torch.manual_seed(42)

def generate_data():
    x = torch.linspace(-10, 10, 100).reshape(-1, 1)
    y_true = 3 * x + 2 + torch.randn(x.shape) * 2
    return x, y_true

def main():
    X, y_true = generate_data()

    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    learning_rate = 0.01
    epochs = 200

    for epoch in range(epochs):
        # Make predictions
        y_prediction = X * w + b

        # Calculate error
        loss = ((y_prediction - y_true) ** 2).mean()

        # Calculate gradients
        loss.backward()

        # Update weight and bias
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # Reset gradients
        w.grad.zero_()
        b.grad.zero_()

        if epoch % 20 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():.3f} | w: {w.item():.3f} | b: {b.item():.3f}")

    print("Final weight:", w.item())
    print("Final bias:", b.item())

if __name__ == "__main__":
    main()
