import pickle

from aalpy.base import SUL


class RNNSul(SUL):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.nn.eval()

    def pre(self):
        self.nn.reset_hidden_state()

    def post(self):
        pass

    def step(self, letter):
        return self.nn.step(letter)


def save_to_file(obj, path):
    pickle.dump(obj, open(f'{path}.pk', "wb"))


def load_from_file(path):
    try:
        with open(f'{path}.pk', "rb") as f:
            return pickle.load(f)
    except IOError:
        return None
