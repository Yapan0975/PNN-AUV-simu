"""
models_bench.py -- digital SOTA-style baselines for the IARA benchmark, on the
standard log-mel front end, plus parameter / MAC accounting so the benchmark
table can place each method on the accuracy-vs-compute axis.

Models (all 2-D log-mel input [B,1,n_mels,T]):
  * LogMelMLP   : flatten -> 2-hidden-layer MLP   (strong MLP, proper feature)
  * CNN2D       : 3-block conv net + GAP           (the classic UATR baseline)
  * ResNetSmall : 3 residual stages + GAP          (the *strong* digital baseline)

count_macs() returns multiply-accumulates per inference (a hardware-agnostic
energy proxy); at a representative 3.1 pJ/MAC for INT8 digital MACs this maps to
joules/inference, to be contrasted with the device-model PNN figure from Phase 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
def _gap(x):
    return x.mean(dim=(2, 3))


class LogMelMLP(nn.Module):
    def __init__(self, n_mels, n_frames, n_classes, hidden=128):
        super().__init__()
        d = n_mels * n_frames
        self.net = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(d),
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class CNN2D(nn.Module):
    def __init__(self, n_classes, ch=(16, 32, 64)):
        super().__init__()
        c0 = 1
        blocks = []
        for c in ch:
            blocks += [nn.Conv2d(c0, c, 3, padding=1), nn.BatchNorm2d(c),
                       nn.ReLU(), nn.MaxPool2d(2)]
            c0 = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(c0, n_classes)

    def forward(self, x):
        return self.head(_gap(self.features(x)))


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = (nn.Sequential() if stride == 1 and cin == cout
                      else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                         nn.BatchNorm2d(cout)))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + self.short(x))


class ResNetSmall(nn.Module):
    """A ResNet-style net (~ResNet-10 footprint) -- the strong digital baseline."""
    def __init__(self, n_classes, widths=(16, 32, 64)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU())
        self.s1 = _ResBlock(widths[0], widths[0], stride=1)
        self.s2 = _ResBlock(widths[0], widths[1], stride=2)
        self.s3 = _ResBlock(widths[1], widths[2], stride=2)
        self.head = nn.Linear(widths[2], n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.s3(self.s2(self.s1(x)))
        return self.head(_gap(x))


# ----------------------------------------------------------------------------
def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def count_macs(model, input_shape):
    """Multiply-accumulates per inference via forward hooks on Conv2d/Linear."""
    macs = [0]
    hooks = []

    def conv_hook(m, inp, out):
        cout = out.shape[1]
        oh, ow = out.shape[2], out.shape[3]
        k = m.kernel_size[0] * m.kernel_size[1]
        cin = m.in_channels // m.groups
        macs[0] += cout * oh * ow * cin * k

    def lin_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    for h in hooks:
        h.remove()
    return int(macs[0])


def build(name, n_mels, n_frames, n_classes):
    if name == "logmel_mlp":
        return LogMelMLP(n_mels, n_frames, n_classes)
    if name == "cnn2d":
        return CNN2D(n_classes)
    if name == "resnet":
        return ResNetSmall(n_classes)
    raise ValueError(name)


if __name__ == "__main__":
    for nm in ("logmel_mlp", "cnn2d", "resnet"):
        m = build(nm, 48, 62, 5)
        print(f"{nm:12s} params {count_params(m):>8d}  "
              f"MACs {count_macs(m, (1, 48, 62)):>10d}")
