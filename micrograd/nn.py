from engine import Value
import random


class Neuron:
    def __init__(self, n):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n)]
        self.b = Value(random.uniform(-1,1))
        self.params = self.w + [self.b]

    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act += wi * xi
        return act.tanh()


class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]
        self.params = [p for neuron in self.neurons for p in neuron.params]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, n_in, n_out):
        size = [n_in] + n_out
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(size)-1)]
        self.params = [p for layer in self.layers for p in layer.params]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SGD:
    def __init__(self, params, lr=0.05):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        for p in self.params:
            p.data += -self.lr * p.grad

