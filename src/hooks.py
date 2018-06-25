import torch.nn as nn
from collections import namedtuple, defaultdict
import torch
import pandas
import numpy
from torch.autograd import Variable
import os
import numpy as np

class AnalysisHook:
    def __init__(self, save_dir, key, analyzer):
        Stat = namedtuple('Stat', ['name', 'func'])

        self.key = key
        self.save_dir = save_dir

        self.analyzer = analyzer

        self.layer_files = {}

    def save_activations_hook(self, module, input, output):
        for idx, activation in enumerate(output):
            activation = self.process_activation(activation)
            if activation is None:
                continue

            if idx not in self.layer_files:
                self.layer_files[idx] = open(os.path.join(self.save_dir, '{}_{}.npy'.format(self.key, idx)), 'wb')

            np.save(self.layer_files[idx], activation.cpu().numpy())

    def process_activation(self, activation):
        if activation is None:
            return None

        if type(activation) is tuple:
            activation = torch.stack(activation, dim=0)

        if type(activation.data) is not torch.cuda.FloatTensor:
            return None

        if activation.dim() == 3 and (activation.size(0) * activation.size(1)) == (self.analyzer.sequence.size(0)):
            # activation: sequence_length x batch_size x hidden_size
            activation = activation.view(self.analyzer.sequence.size(0), -1)
        elif activation.size(0) != self.analyzer.sequence.size(0):
            return None
        # activation: (sequence_length * batch_size) x hidden_size

        return activation.data

    def register_hook(self, module):
        self.handle = module.register_forward_hook(self.save_activations_hook)

    def remove_hook(self):
        for idx, file in self.layer_files.items():
            file.close()
        if self.handle is not None:
            self.handle.remove()
        self.layer_files = {}

class NetworkSubspaceConstructor:
    def __init__(self, model, save_dir, normalize=False):
        self.model = model
        self.hooks = {}
        self.save_dir = save_dir
        self.normalize = normalize

        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)

    @staticmethod
    def module_output_size(module):
        # return the size of the final parameters in the module,
        # or 0 if there are no parameters
        output_size = 0
        for key, parameter in module.named_parameters():
            if key.find('weight') < 0:
                continue
            output_size = parameter.size(-1)
        return output_size

    def set_word_sequence(self, targets):
        self.sequence = targets.data.view(-1)

    def add_hooks_recursively(self, parent_module: nn.Module, prefix=''):
        # add hooks to the modules in a network recursively
        for module_key, module in parent_module.named_children():
            output_size = self.module_output_size(module)
            if output_size == 0:
                continue
            module_key = prefix + module_key
            self.hooks[module_key] = AnalysisHook(self.save_dir, module_key, self)

            self.hooks[module_key].register_hook(module)
            self.add_hooks_recursively(module, prefix=module_key)

    def add_hooks_to_model(self):
        self.add_hooks_recursively(self.model)

    def remove_hooks(self):
        for key, hook in self.hooks.items():
            hook.remove_hook()
