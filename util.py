import pickle
from random import choice, randint

import torch
from aalpy.base import SUL


class RNNSul(SUL):
    def __init__(self, nn, clustering_fun=None):
        super().__init__()
        self.clustering_fun = clustering_fun
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


def conformance_test(nn_model, automaton, n_tests=10000, min_test_len=16, max_test_len=30):
    sul = RNNSul(nn_model)
    input_al = automaton.get_input_alphabet()

    cex_counter = 0
    for _ in range(n_tests):
        tc = [choice(input_al) for _ in range(randint(min_test_len, max_test_len))]

        sul.pre()
        automaton.reset_to_initial()
        for i in tc:
            o_sul = sul.step(i)
            o_aut = automaton.step(i)
            if o_sul != o_aut:
                cex_counter += 1
                break
        sul.post()

    print(f'Conformance Testing with {n_tests} Random Strings Found {cex_counter} counterexamples.')
    return cex_counter / n_tests
