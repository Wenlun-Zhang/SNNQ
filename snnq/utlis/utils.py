import torch.nn as nn
from snnq.modules.modules import *
import os
import yaml
from easydict import EasyDict
from snnq.quantization.quantized_module import QuantizedLayer
from snnq.quantization.quant_model import specials


def isActivation(name):
    if 'relu' in name.lower() or 'qcfs' in name.lower():
        return True
    return False


def replace_MPLayer_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_MPLayer_by_neuron(module)
        if module.__class__.__name__ == 'MPLayer':
            model._modules[name] = IFNeuron(scale=module.v_threshold)
    return model


def replace_activation_by_MPLayer(model, presim_len, sim_len):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_MPLayer(module, presim_len, sim_len)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = MPLayer(v_threshold=module.up.item(), presim_len=presim_len, sim_len=sim_len)
    return model


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model


def replace_activation_by_floor(model, t):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = QCFS(up=8., t=t)
    return model


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def error(info):
    print(info)
    exit(1)
    

class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


def get_cali_data(train_loader, num_samples):
    cali_data = []
    for batch in train_loader:
        cali_data.append(batch[0])
        if len(cali_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(cali_data, dim=0)[:num_samples]


def quantize_model(model, config_quant):

    def replace_module(module, w_qconfig):
        for name, child in module.named_children():
            if type(child) in specials:
                setattr(module, name, specials[type(child)](child, w_qconfig))
            elif isinstance(child, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child, None, w_qconfig))
            else:
                replace_module(child, w_qconfig)

    if type(model) in specials:
        return specials[type(model)](model, config_quant.w_qconfig)

    replace_module(model, config_quant.w_qconfig)

    return model


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config
