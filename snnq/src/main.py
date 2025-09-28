import sys
sys.path.append("/home/zhang/Project/SNNQ")
from snnq.utlis.NetworkFunction import *
import argparse
from snnq.utlis.dataprocess import PreProcess_Cifar10, PreProcess_Cifar100, PreProcess_ImageNet
from model.ResNet import *
from model.VGG import *
import torch
import random
import os
import numpy as np
import copy
import time
from snnq.optim.recon import reconstruction
from snnq.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from snnq.quantization.quantized_module import QuantizedLayer, QuantizedBlock
from snnq.quantization.observer import ObserverBase
from snnq.utlis.utils import parse_config, get_cali_data, quantize_model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset name')
    parser.add_argument('--datadir', type=str, default='/home/dataset/SNNQ', help='Directory where the dataset is saved')
    parser.add_argument('--savedir', type=str, default='/home/zhang/Model/', help='Directory where the model is saved')
    parser.add_argument('--load_model_name', type=str, default='None', help='The name of the loaded ANN model')
    parser.add_argument('--trainann_epochs', type=int, default=200, help='Training Epochs of ANNs')
    parser.add_argument('--activation_floor', type=str, default='QCFS', help='ANN activation modules')
    parser.add_argument('--net_arch', type=str, default='vgg16', help='Network Architecture')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size')
    parser.add_argument('--L', type=int, default=4, help='Quantization level of QCFS')
    parser.add_argument('--run_step', type=int, default=128, help='Time step to run the SNN')
    parser.add_argument('--test_step', type=int, default=4, help='Time step to test the SNN')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--direct_training', action='store_true', default=False)
    parser.add_argument('--train_dir', type=str, default='/home/dataset/ImageNet/train', help='Directory where the ImageNet train dataset is saved')
    parser.add_argument('--test_dir', type=str, default='/home/dataset/ImageNet/val', help='Directory where the ImageNet test dataset is saved')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')

    parser.add_argument('--q_config', type=str, default='./exp/config.yaml', help='Quantization config files')
    parser.add_argument('--fp_model', action='store_true', default=False, help='Run with FP model')
    parser.add_argument('--bwr', action='store_true', default=False, help='Optimize quantization parameters via model reconstruction')
    parser.add_argument('--tnp', action='store_true', default=False, help='Activate test-time pruning')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    
    torch.backends.cudnn.benchmark = True
    _seed_ = args.seed
    random.seed(_seed_)
    os.environ['PYTHONHASHSEED'] = str(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed_)

    cls = 10
    cap_dataset = 10000
    
    if args.dataset == 'CIFAR10':
        cls = 10
    elif args.dataset == 'CIFAR100':
        cls = 100
    elif args.dataset == 'ImageNet':
        cls = 1000
        cap_dataset = 50000
    
    
    if args.net_arch == 'resnet20':
        model = resnet20(num_classes=cls)
    elif args.net_arch == 'resnet18':
        model = resnet18(num_classes=cls)
    elif args.net_arch == 'resnet34':
        model = resnet34(num_classes=cls)
    elif args.net_arch == 'vgg16':
        model = vgg16(num_classes=cls)
    else:
        error('unable to find model ' + args.arch)
    
    model = replace_maxpool2d_by_avgpool2d(model)
    
    if args.activation_floor == 'QCFS':
        model = replace_activation_by_floor(model, args.L)
    else:
        error('unable to find activation floor: ' + args.activation_floor)
    
    if args.dataset == 'CIFAR10':
        train, test = PreProcess_Cifar10(args.datadir, args.batchsize)
    elif args.dataset == 'CIFAR100':
        train, test = PreProcess_Cifar100(args.datadir, args.batchsize)
    elif args.dataset == 'ImageNet':
        train, test = PreProcess_ImageNet(args.datadir, args.batchsize, train_dir=args.train_dir, test_dir=args.test_dir)
    else:
        error('unable to find dataset ' + args.dataset)


    if args.load_model_name != 'None':
        print(f'=== Load Pretrained ANNs ===')
        model.load_state_dict(torch.load(args.load_model_name + '.pth'))
    if args.direct_training is True:
        print(f'=== Start Training ANNs ===')
        save_name = args.savedir + args.activation_floor + '_' + args.dataset + '_' + args.net_arch + '_L' + str(args.L) + '.pth'
        model = train_ann(train, test, model, epochs=args.trainann_epochs, lr=args.lr, wd=args.wd, device=args.device, save_name=save_name)

    
    print(f'=== ANNs accuracy after the first training stage ===')
    acc = eval_ann(test, model, args.device)
    print(f'Pretrained ANN Accuracy : {acc / cap_dataset}')

    if not args.fp_model:
        print(f'=== Start quantization ===')
        q_config = parse_config(args.q_config)
        cali_data = get_cali_data(train, q_config.calibrate)
        model = quantize_model(model, q_config)
        model.cuda()
        model.eval()
        fp_model = copy.deepcopy(model)
        disable_all(fp_model)
        for name, module in model.named_modules():
            if isinstance(module, ObserverBase):
                module.set_name(name)

        # calibrate first
        with torch.no_grad():
            st = time.time()
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            model(cali_data[:32].cuda())
            ed = time.time()
            print('the calibration time is {}'.format(ed - st))

        if args.bwr:
            print(f'=== Start reconstruction optimization ===')
            enable_quantization(model)

            def recon_model(module: nn.Module, fp_module: nn.Module):
                for name, child_module in module.named_children():
                    if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                        print('begin reconstruction for module:\n{}'.format(str(child_module)))
                        reconstruction(model, fp_model, child_module, getattr(fp_module, name), cali_data, q_config.recon)
                    else:
                        recon_model(child_module, getattr(fp_module, name))
            # Start reconstruction
            recon_model(model, fp_model)
        enable_quantization(model)

        if not args.tnp:
            args.test_step = 0

    replace_activation_by_MPLayer(model, presim_len=args.test_step, sim_len=args.run_step)

    print(f'=== SNNs accuracy after the SRP stage ===')

    if args.test_step > 0:
        new_acc = mp_test(test, model, net_arch=args.net_arch, presim_len=args.test_step, sim_len=args.run_step, device=args.device)
    else:
        replace_MPLayer_by_neuron(model)
        new_acc = eval_snn(test, model, sim_len=args.run_step, device=args.device)

    t = 1
    while t < args.run_step:
        print(f'time step {t}, Accuracy = {(new_acc[t-1] / cap_dataset):.4f}')
        t *= 2
    print(f'time step {args.run_step}, Accuracy = {(new_acc[args.run_step-1] / cap_dataset):.4f}')
