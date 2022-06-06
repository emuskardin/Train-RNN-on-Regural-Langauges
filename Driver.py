import torch.optim as optim
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.utils import load_automaton_from_file

from AutomataDataGen import generate_data_from_automaton, AutomatonDataset, get_tomita, get_coffee_machine, \
    get_ssh, get_angluin

from RNN import get_model, Optimization
from util import RNNSul

# Some preloaded automata
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

# 'cpu' to train on cpu, 'gpu' to train on cuda:0
device = 'cpu'

# to use preloaded automata
exp_name = 'tomita3'
automaton = experiments[exp_name]

# Number of training and validation samples
num_training_samples = 50000
num_val_samples = 4000

# Do not learn the original automaton, but a mapping of sequence of inputs to reached state
# Each state is represented by unique state_id
classify_states = False
exp_name = exp_name if not classify_states else exp_name + '_states'

# Generate training and validation data
training_data, input_al, output_al = generate_data_from_automaton(automaton, num_training_samples,
                                                                  classify_states=classify_states)

# training_data, input_al, output_al = get_characterizing_set_training_data(automaton, prefix_closed=True)
validation_data, _, _ = generate_data_from_automaton(automaton, num_val_samples, classify_states=classify_states)

# Setup RNN parameters
model_type = 'gru'
activation_fun = 'relu'  # note that activation_fun value is irrelevant for GRU and LSTM
input_dim = len(input_al)
output_dim = len(output_al)
hidden_dim = 20
layer_dim = 2
batch_size = 128
dropout = 0  # 0.1 if layer_dim > 1 else 0
n_epochs = 100
optimizer = optim.Adam
learning_rate = 0.0005
weight_decay = 1e-6
early_stop = True  # Stop training if loss is smaller than small threshold for few epochs

data_handler = AutomatonDataset(input_al, output_al, batch_size)

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

# This will train the RNN
# If trained model with same parametrization exists, it will be loaded unless load flag is set to False
opt.train(train, val, n_epochs=n_epochs, exp_name=exp_name, early_stop=early_stop, save=True, load=True)

# Extraction process - from RNN to regular language
sul = RNNSul(model)

eq_oracle = RandomWMethodEqOracle(input_al, sul, walks_per_state=200, walk_len=10)

# currently only mealy is supported
model = run_Lstar(input_al, sul, eq_oracle, 'mealy')

model.visualize()
