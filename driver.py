import torch
import torch.optim as optim
from random import shuffle

from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import load_automaton_from_file

from automata_data_generation import generate_data_from_automaton, get_tomita, \
    get_coffee_machine, get_ssh, get_angluin, AutomatonDataset
from RNN import get_model, Optimization
from util import conformance_test, RNNSul

experiments = {'tomita3': get_tomita(3),
               'tomita5': get_tomita(5),
               'tomita7': get_tomita(7),
               'coffee': get_coffee_machine(),
               'ssh': get_ssh(),
               'angluin': get_angluin(),
               'regex': load_automaton_from_file('automata/regex_paper.dot', automaton_type='dfa',
                                                 compute_prefixes=True),
               'tree': load_automaton_from_file('automata/tree.dot', automaton_type='dfa',
                                                compute_prefixes=True),
               'last_a': load_automaton_from_file('automata/last_a.dot', automaton_type='dfa',
                                                  compute_prefixes=True)}

device = None  # for auto detection

exp_name = 'coffee'
automaton = experiments[exp_name]

# Number of training and validation samples
num_training_samples = 30000
num_val_samples = 4000

# Generate training and validation data
# lens=(1,2,3,4,6,8,10,12,15)
automaton_data, input_al, output_al = generate_data_from_automaton(automaton, num_training_samples + num_val_samples,)

shuffle(automaton_data)

training_data, validation_data = automaton_data[:num_training_samples], automaton_data[num_training_samples:]

# Setup RNN parameters
model_type = 'gru'
activation_fun = 'relu'  # note that activation_fun value is irrelevant for GRU and LSTM
input_dim = len(input_al)
output_dim = len(output_al)
hidden_dim = 20
layer_dim = 2
batch_size = 64
dropout = 0  # 0.1 if layer_dim > 1 else 0
n_epochs = 100
optimizer = optim.Adam
learning_rate = 0.0005
weight_decay = 1e-6
early_stop = True  # Stop training if loss is smaller than small threshold for few epochs

data_handler = AutomatonDataset(input_al, output_al, batch_size, device=device)

train, val = data_handler.create_dataset(training_data), data_handler.create_dataset(validation_data)

model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'nonlinearity': activation_fun,
                'dropout_prob': dropout,
                'data_handler': data_handler,
                'device': device}

model = get_model(model_type, model_params)

optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, optimizer=optimizer, device=device)

model.get_model_name(exp_name)
process_hs_fun = 'flatten_lstm' if model_type == 'lstm' else 'flatten'

# This will train the RNN
# If trained model with same parametrization exists, it will be loaded unless load flag is set to False
opt.train(train, val, n_epochs=n_epochs, exp_name=exp_name, early_stop=early_stop, save=True, load=True)

# disable all gradient computations to speed up execution
torch.no_grad()

# check the RNN for accuracy on randomly generated data
conformance_test(model, automaton, n_tests=1000, max_test_len=30)

# wrap RNN in AALpy's SUL interface
sul = RNNSul(model)
# this is a weak eq. oracle with weak configuration
eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=1000, max_walk_len=10)

learned_model = run_Lstar(input_al, sul, eq_oracle, 'mealy')
learned_model.visualize()
