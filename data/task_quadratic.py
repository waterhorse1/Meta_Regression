import numpy as np
import torch


class quadratic:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self):
        self.num_inputs = 1
        self.num_outputs = 1

        #self.amplitude_range = [0.0, 0.15]
        #self.c = [-3, 3]
        #self.b = [-3, 3]
        #self.negative = [1,-1]
        
        self.amplitude_range = [0.1, 0.11]
        self.c = [1, 1.1]
        self.b = [1, 1.1]
        self.negative = [1]

        self.input_range = [-5, 5]

    def get_input_range(self, size=100):
        return torch.linspace(self.input_range[0], self.input_range[1], steps=size).unsqueeze(1)

    def sample_inputs(self, batch_size, *args, **kwargs):
        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return inputs

    def sample_task(self):
        negative = np.random.choice(self.negative)
        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1]) * negative
        b = np.random.uniform(self.b[0], self.b[1])
        c = np.random.uniform(self.c[0], self.c[1])
        return self.get_target_function(amplitude, b, c)

    @staticmethod
    def get_target_function(amplitude, b, c):
        def target_function(x):
            if isinstance(x, torch.Tensor):
                return (x - c) ** 2 * amplitude + b
            else:
                return (x - c) ** 2 * amplitude + b

        return target_function

    def sample_tasks(self, num_tasks, return_specs=False):

        amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], num_tasks)
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], num_tasks)

        target_functions = []
        for i in range(num_tasks):
            target_functions.append(self.get_target_function(amplitude[i], phase[i]))

        if return_specs:
            return target_functions, amplitude, phase
        else:
            return target_functions

