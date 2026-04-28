import torch
import os
import torch.nn.functional as F


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
NAMES_PATH = os.path.join(PROJECT_ROOT,"name_generator","names.txt")


with open(NAMES_PATH,"r") as f:
    NAMES = f.read().splitlines()

LETTERS = sorted(list(set("".join(NAMES))))

class Bigram:
    def __init__(self):
        self.N = torch.zeros((27,27))
        self.stoi = {letter:i+1 for i, letter in enumerate(LETTERS)}
        self.stoi["."] = 0
        self.itos = {value:key for key, value in self.stoi.items()}
        self.P = None

    def load_bigram(self):
        for name in NAMES:
            chs = ["."] + list(name) + ["."]
            for char1, char2 in zip(chs, chs[1:]):
                i, j = self.stoi[char1], self.stoi[char2]
                self.N[i,j] += 1

    def normalize(self):
        self.P = (self.N+1).float()
        self.P /= self.P.sum(1, keepdim=True)

    def generate(self, n):
        generator = torch.Generator()
        for _ in range(n):
            i = 0
            out_names = []
            while True:
                p = self.P[i]
                i = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
                out_names.append(self.itos[i])
                if i == 0:
                    break
            print(''.join(out_names))


class NNBigram:
    def __init__(self):
        self.x = []
        self.y = []
        self.w = None
        self.stoi = {letter:i+1 for i, letter in enumerate(LETTERS)}
        self.stoi["."] = 0
        self.itos = {value:key for key, value in self.stoi.items()}
        self._load_data()

    def _load_data(self):
        for name in NAMES:
            chs = ["."] + list(name) + ["."]
            for x, y in zip(chs, chs[1:]):
                x_int, y_int = self.stoi[x], self.stoi[y]
                self.x.append(x_int)
                self.y.append(y_int)

        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)
        self.x = F.one_hot(self.x, num_classes=27).float()
        self.w = torch.randn((27, 27), requires_grad=True)

    def train(self, epochs, lr):
        for i in range(epochs):
            self.w.grad = None
            # Forward
            logits = self.x @ self.w
            probs = logits.exp()
            softmax = probs / torch.sum(probs, dim=1, keepdim=True)
            loss = -softmax[torch.arange(self.x.shape[0]),self.y].log().mean() + 0.01*(self.w**2).mean()

            # Backward
            loss.backward()
            self.w.data -= lr * self.w.grad

            #print(f"{loss.item():.4f}")

    def generate(self, n):
        g = torch.Generator()
        for i in range(n):
            out = []
            i = 0
            while True:
                current = F.one_hot(torch.tensor([i]), num_classes=27).float()
                logits = current @ self.w
                probs = logits.exp()
                softmax = probs / torch.sum(probs, dim=1, keepdim=True)
                i = torch.multinomial(softmax, num_samples=1, replacement=True, generator=g).item()
                out.append(self.itos[i])
                if i == 0:
                    break
            print(''.join(out))