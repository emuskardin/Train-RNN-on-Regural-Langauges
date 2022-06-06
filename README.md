# Train RNNs to recognize regular languages

This repository contains minimal code which can be used to train RNN (plain, GRU, LSTM) to recognize regular languages.
It also provides a method of extracting regular languages form trained RNNs using active automata learning.

## Install 
```
pip install -r recquirements.txt
```
To run experiments with CUDA do:
```
pip3 install torch==1.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
To visualize finite state machines, ensure that Graphviz is installed on your system.

## Run Training
To run the training, simply configure to your wishes and run the `Driver.py` script.

In it, you can train a RNN that will try to learn the input-output behaviour of the regular language.
You can change the regular language that is to be learned, RNN type, and other (hyper)parameters. 

## Repo Structure
- Driver.py - file in which RNN is trained to recognize a regular language, and an automaton is extracted from the RNN
- RNN.py - RNN class, including plain RNN with 2 activation functions, as well as gru and lstm RNN's
- AutomataDataGen.py - create data and batches from reg. language
- util.py - util functions

