# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# from torch.autograd import *

# import torchvision
# import torchvision.transforms as transforms

# import os
# import argparse

# from resnet import *

# import pickle


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# lr = 0.01

# if torch.cuda.is_available():
#     torch.cuda.set_device(0)  # Selects the second GPU
#     print("PyTorch is using GPU.")
#     print("Number of GPUs available:", torch.cuda.device_count())
#     print("Current GPU:", torch.cuda.current_device())
# else:
#     print("PyTorch is using CPU.")

import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='device range [0,ngpu-1]')


args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    # make only devices indexed by #gpu_id visible
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)