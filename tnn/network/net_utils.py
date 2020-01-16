import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
# from torch.legacy.nn.SpatialUpSamplingNearest import SpatialUpSamplingNearest
from torch.nn import UpsamplingNearest2d

import numpy as np


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.pointwise(x)
        x = F.relu(self.bn2(x), inplace=True)
        return x


class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bias=False):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        # momentum = 0.05 if self.training else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_CReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, same_padding=False, bias=False):
        super(Conv2d_CReLU, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        # momentum = 0.05 if self.training else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.scale = Scale(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        x = self.scale(x)
        return self.relu(x)


class Scale(nn.Module):
    def __init__(self, in_channels):
        super(Scale, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        return x * self.alpha.expand_as(x) + self.beta.expand_as(x)


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class UpSamplingFunction(Function):
    def __init__(self, scale):
        self.scale = scale
        self.up = UpsamplingNearest2d(scale_factor=scale)

    def forward(self, x):
        self.save_for_backward(x)

        if x.is_cuda:
            self.up.cuda()
        return self.up.updateOutput(x)

    def backward(self, grad_output):
        return self.up.updateGradInput(self.saved_tensors[0], grad_output)


class UpSampling(nn.Module):
    def __init__(self, scale):
        super(UpSampling, self).__init__()
        self.scale = scale

    def forward(self, x):
        return UpSamplingFunction(self.scale)(x)


def str_is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def save_net(fname, net, epoch=-1, lr=-1):
    import h5py
    with h5py.File(fname, mode='w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

        h5f.attrs['epoch'] = epoch
        h5f.attrs['lr'] = lr


def load_net(fname, net, prefix='', stacked=False):
    import h5py
    with h5py.File(fname, mode='r') as h5f:
        h5f_is_module = True
        for k in h5f.keys():
            if not str(k).startswith('module.'):
                h5f_is_module = False
                break
        if prefix == '' and not isinstance(net, nn.DataParallel) and h5f_is_module:
            prefix = 'module.'

        for k, v in net.state_dict().items():
            k = prefix + k
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                if v.size() != param.size():
                    print('inconsistent shape: {}, {}'.format(v.size(), param.size()))
                else:
                    v.copy_(param)
            else:
                # find if stacked
                if stacked:
                    root_name = k.split('.')[0]
                    idx = root_name.rfind('_')
                    if idx >= 0 and str_is_int(root_name[idx+1:]):
                        try_k = root_name[:idx] + k[k.find('.'):]
                        if try_k in h5f:
                            print('stacked: {}, {}'.format(k, try_k))
                            param = torch.from_numpy(np.asarray(h5f[try_k]))
                            if v.size() != param.size():
                                print('inconsistent shape: {}, {}'.format(v.size(), param.size()))
                            else:
                                v.copy_(param)
                            continue

                print('no layer: {}'.format(k))

        epoch = h5f.attrs['epoch'] if 'epoch' in h5f.attrs else -1
        lr = h5f.attrs['lr'] if 'lr' in h5f.attrs else -1.

        return epoch, lr


def plot_graph(top_var, fname):
    """
    Plot the graph. Make sure that require_grad=True and volatile=False
    :param top_var: network output Varibale
    :param fname: file name
    :return: None
    """
    from graphviz import Digraph
    import pydot
    dot = Digraph(comment='LRP',
                  node_attr={'style': 'filled', 'shape': 'box'})
    # , 'fillcolor': 'lightblue'})

    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                dot.node(str(id(var)), str(var.size()), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), type(var).__name__)
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])

    add_nodes(top_var.creator)
    dot.save(fname)
    print(fname)
    (graph,) = pydot.graph_from_dot_file(fname)
    graph.write_png('{}.png'.format(fname))


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False, device_id=None):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile)
    if is_cuda:
        v = v.cuda(device_id)
    return v


def variable_to_np_tf(x):
    return x.data.cpu().numpy().transpose([0, 2, 3, 1])


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)
