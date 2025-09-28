from .fake_quant import QuantizeBase


def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_quantization(model, quantizer_type='fake_quant'):
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            submodule.disable_observer()
            submodule.enable_fake_quant()


def disable_all(model):
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            submodule.disable_observer()
            submodule.disable_fake_quant()