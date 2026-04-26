import torch
import os


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

