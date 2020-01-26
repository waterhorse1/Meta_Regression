import numpy as np
import torch
#from data.task_sine import sine
from data.task_test_sine import sine
from data.task_norm import norm
from data.task_linear import linear
#from data.task_quadratic import quadratic
from data.task_test_quadratic import quadratic
from data.task_tanh import tanh
from data.task_cubic import cubic


class multi:
    """
    Same regression task as in Finn et al. 2017 (MAML)
    """

    def __init__(self):
        self.num_inputs = 1
        self.num_outputs = 1
        self.input_range = [-5, 5]
        self.task = [sine(),quadratic(),linear(),cubic()]#sine(), cubic()]#】,quadratic()]
        self.task_num = len(self.task)

    def get_input_range(self, size=100):
        return torch.linspace(self.input_range[0], self.input_range[1], steps=size).unsqueeze(1)

    def sample_inputs(self, batch_size, *args, **kwargs):
        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        return inputs

    def sample_task(self, return_type = False):
        num = np.random.choice(np.array(range(self.task_num)))
        if not return_type:
            return self.task[num].sample_task()
        else:
            return self.task[num].sample_task(), num
            

    def sample_tasks(self, num_tasks, return_type = False):        
        if not return_type:
            target_functions = []
            for i in range(num_tasks):
                target_functions.append(self.sample_task())
            return target_functions
        else:
            target_functions = []
            num = []
            for i in range(num_tasks):
                func, n = self.sample_task(True)
                target_functions.append(func)
                num.append(n)
            return target_functions, num
            
    '''
    def sample_datapoints(self, batch_size):
        """
        Sample random input/output pairs (e.g. for training an orcale)
        :param batch_size:
        :return:
        """

        amplitudes = torch.Tensor(np.random.uniform(self.amplitude_range[0], self.amplitude_range[1], batch_size))
        phases = torch.Tensor(np.random.uniform(self.phase_range[0], self.phase_range[1], batch_size))

        inputs = torch.rand((batch_size, self.num_inputs))
        inputs = inputs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
        inputs = inputs.view(-1)

        outputs = torch.sin(inputs - phases) * amplitudes
        outputs = outputs.unsqueeze(1)

        return torch.stack((inputs, amplitudes, phases)).t(), outputs
    '''
