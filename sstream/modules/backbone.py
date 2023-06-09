import functools
from abc import abstractmethod
import typing as ty

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torchvision.models import resnet18




class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier: nn.Module
        self.hooks = []
        self.intermediate_feats = {}
        self.intermediate_grads = {}

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, compute_feats=False):
        if compute_feats:
            self.add_hooks()

        logits = self._forward(x)

        if compute_feats:
            return logits
        return logits

    @abstractmethod
    def _forward(self, x):
        return

    def clean_hooks(self):
        [hook.remove() for hook in self.hooks]
        self.hooks = []
        self.intermediate_feats = {}

    def add_hooks(self):
        def interm_feats(module, input, output, name):
            self.intermediate_feats[name] = output.clone().detach()

        def interm_grads(module, input, output, name):

            self.intermediate_grads[name] = [
                _input.clone().detach() for _input in input if _input is not None
            ]

        if len(self.hooks) == 0:
            for n, m in self.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    forward_hook = functools.partial(interm_feats, name=n)
                    hook = m.register_forward_hook(forward_hook)
                    self.hooks.append(hook)
                    back_hook = functools.partial(interm_grads, name=n)
                    hook = m.register_full_backward_hook(back_hook)
                    self.hooks.append(hook)

    def get_params(self) -> torch.Tensor:
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        return torch.cat(self.get_grads_list())

    def get_grads_list(self) -> list[torch.Tensor]:
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return grads



class ResNetModel(Backbone):
    def __init__(self, output_dim):
        super().__init__()
        self.model = resnet18()
        # MODEL SUR-GERY
        self.model.conv1 = nn.Identity()
        self.model.fc = nn.Identity()
        self.classifier = nn.Linear(512, output_dim)

    def _forward(self, x):
        # shape [batch, 64, h, w]
        h = self.model(x)
        logits = self.classifier(h)
        return logits


class ResLinear(nn.Module):
    def __init__(self, in_planes: int, planes: int, expansion: int = 1) -> None:

        super().__init__()
        self.fc1 = nn.Linear(in_planes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.out_planes = expansion * planes
        self.fc2 = nn.Linear(planes, self.out_planes)
        self.bn2 = nn.BatchNorm1d(self.out_planes)

        self.shortcut = nn.Sequential(
            nn.Linear(in_planes, self.out_planes, bias=False),
            nn.BatchNorm1d(self.out_planes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResMLP(Backbone):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.input_size = input_dim[0]
        self.output_size = output_dim

        hidden_dim = 32
        expansion = 2
        self.input_projection = nn.Linear(self.input_size, hidden_dim)
        self.nets = nn.ModuleList()
        self.nets.append(ResLinear(hidden_dim, hidden_dim, expansion=expansion))
        self.nets.append(
            ResLinear(self.nets[-1].out_planes, hidden_dim, expansion=expansion)
        )

        self.classifier = nn.Linear(self.nets[-1].out_planes, self.output_size)
        self.dropout = nn.Dropout(p=0.15)
        self.apply(self._init_weights)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1, -1)
        feats = self.input_projection(x)
        for net in self.nets:
            feats = net(feats)
            feats = self.dropout(feats)

        out = self.classifier(feats)
        return out


def make_backbone(
    name: ty.Literal["linear", "resnet18"], input_dim, output_dim
) -> Backbone:
    if name == "linear":
        return ResMLP(input_dim=input_dim, output_dim=output_dim)
    elif name == "resnet18":
        return ResNetModel(output_dim=output_dim)
    else:
        raise NotImplementedError("Backbone not implemented")
