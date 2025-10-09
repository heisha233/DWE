import sys
from models import *
import torch.nn as nn

def get_network(network, in_channels, num_classes, **kwargs):

    # 2d networks
    if network == 'DWE':
        net = DWE(in_channels, num_classes)
    elif network == 'xnet':
        net = XNet(in_channels, num_classes)
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
