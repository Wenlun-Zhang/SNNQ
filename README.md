# SNNQ
Post-Training Quantization Framework in "SNNQ: Post-Training Quantization Towards Ultra Low-Bit and Fast Spiking Neural Networks".

## 1. Run Experiments

Please use the following command to run SNNQ quantization:

```
python ./snnq/src/main.py --bwr --tnp \
--dataset <DATASET> \
--load_model_name <MODEL_PTH> \
--net_arch <MODEL_NAME>
```

Here, ``<DATASET>`` is the name of dataset, ``<MODEL_PTH>`` is the pretrained ANN model path, and ``<MODEL_NAME>`` is the model name.

Also, please configure the quantization setup in `./exp/config.yaml` before you run the experiment.

For example, to perform VGG16 with CIFAR-10, use the following command:

```
python ./snnq/src/main.py --bwr --tnp \
--dataset CIFAR10 \
--net_arch vgg16
```

## 2. Download Model Weights

We provide the pretrained ANN models in this paper to reproduce the experimental results:

| Model     | Dataset     | Download                                                                                    |
|-----------|-------------|---------------------------------------------------------------------------------------------|
| VGG-16    | CIFAR-10    | [Link](https://drive.google.com/file/d/1K09XhYc5RVJtCrvrQbsQB9fG0uC9vqUx/view?usp=sharing)  |
| ResNet-18 | CIFAR-10    | [Link](https://drive.google.com/file/d/1slRs3Q3HtXKKGeJZTYy2mLGmM03GzAvw/view?usp=sharing)  |
| VGG-16    | CIFAR-100   | [Link](https://drive.google.com/file/d/1L1nAc5JEjx2HvUQVJksClfOFlr2POKWh/view?usp=sharing)  |
| ResNet-18 | CIFAR-100   | [Link](https://drive.google.com/file/d/16yuEn2lnTQHcr-FfeQz-bW0WYNPa3WDZ/view?usp=sharing)  |
| VGG-16    | ImageNet-1K | [Link](https://drive.google.com/file/d/1-qMk_3K_G4S2eu9kPyQ4hX4xQcJqKyN5/view?usp=sharing)  |
| ResNet-18 | ImageNet-1K | [Link](https://drive.google.com/file/d/1OVyu9CDkTkI7IMVqoFukBMqqZcwQ-0SW/view?usp=sharing)  |

## 3. Abstract

Spiking Neural Networks (SNNs) are promising for energy-efficient neuromorphic computing but face critical challenges when deployed on resource-constrained devices. Post-Training Quantization (PTQ) provides an attractive solution by avoiding costly retraining, yet existing PTQ frameworks for SNNs struggle with severe accuracy degradation at ultra low-bit settings and long inference windows. In this paper, we propose SNNQ, a PTQ framework that jointly addresses these issues. SNNQ introduces Block-Wise Weight Rounding to optimize rounding factors through block-level reconstruction, ensuring robust performance under low-bit quantization. Furthermore, we develop Test-Time Neuron Pruning, which leverages short calibration runs to prune noisy neurons and significantly reduce inference time steps. Experiments on CIFAR-10, CIFAR-100, and ImageNet-1K demonstrate that SNNQ consistently outperforms prior PTQ methods, achieving accurate 2-bit quantization with no more than 8 time steps. These results highlight the potential of SNNQ for enabling ultra low-bit, low-latency, and energy-efficient SNN deployment.

## Acknowledgments
Our work is built upon [QDrop](https://github.com/wimh966/QDrop) and [ANN2SNN_SRP](https://github.com/hzc1208/ANN2SNN_SRP). We thank for their pioneering work and create a nice baseline for quantization of SNN.
