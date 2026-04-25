from nn import MLP, SGD


xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]


model = MLP(3, [4, 4, 1])
optimizer = SGD(model.params, lr=0.05)

epochs = 20

for epoch in range(epochs):
    ypred = [model(x) for x in xs]

    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.data}")
