'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
import torch
from PIL import Image
import os


def get_method_here(model_name, weights_path):
    if False:
        pass
    else:
        model_name = 'Corvi_pretrain'
        model_path = os.path.join(weights_path, model_name + '.pth')
        arch = 'res50stride1'
        norm_type = 'resnet'
        patch_size = None

    return model_name, model_path, arch, norm_type, patch_size


def rule_minmax(x):
    x = x.reshape(x.shape[0], 1, -1)
    sm = torch.mean(x, -1)
    su = torch.max(x, -1)[0]
    sd = torch.min(x, -1)[0]
    assert sm.shape == su.shape
    return torch.where(sm <= 0, sd, su)


def rule_trim(x, th=np.log(0.8), tl=np.log(0.2)):
    x = torch.nn.functional.logsigmoid(x)
    x = x.view(x.shape[0], 1, -1)
    a = torch.median(x, -1)[0]
    su = torch.sum(torch.where(x >= th, x, torch.zeros_like(x)
                               ), -1) / torch.sum(x >= th, -1)
    sd = torch.sum(torch.where(x <= tl, x, torch.zeros_like(x)
                               ), -1) / torch.sum(x <= tl, -1)
    x = torch.mean(x, -1)
    x = torch.where(a >= th, su, x)
    x = torch.where(a <= tl, sd, x)
    return x


dict_rule = {
    'avg': lambda x: torch.mean(x, (-2, -1)),
    'max': lambda x: torch.quantile(x.reshape(x.shape[0], x.shape[1], -1), 1.0, dim=-1),
    'perc97': lambda x: torch.quantile(x.reshape(x.shape[0], x.shape[1], -1), 0.97, dim=-1),
    'minmax': rule_minmax,
    'trim': rule_trim,
    'lss': lambda x: torch.logsumexp(torch.nn.functional.logsigmoid(x), (-2, -1)),
}


def avpool(x, s):
    import scipy.ndimage as ndi
    h = s//2
    return ndi.uniform_filter(x, (1, s, s), mode='constant')[:, h:1-h, h:1-h]


def def_size_avg(arch):
    if arch == 'res50':
        return 8
    elif arch == 'res50stride1':
        return 8
    else:
        assert False


def def_model(arch, model_path, localize=False):
    import torch

    if arch == 'res50':
        from networks.resnet import resnet50
        model = resnet50(num_classes=1)
    elif arch == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 1)
    elif arch == 'res50stride1':
        import networks.resnet_mod as resnet_mod
        model = resnet_mod.resnet50(num_classes=1, gap_size=1, stride0=1)
    else:
        print(arch)
        assert False

    assert localize is False

    if model_path == 'None':
        Warning('model_path is None! No weights loading in eval.py')
    else:
        dat = torch.load(model_path, map_location='cpu')
        if 'model' in dat:
            if ('module._conv_stem.weight' in dat['model']) or ('module.fc.fc1.weight' in dat['model']) or ('module.fc.weight' in dat['model']):
                model.load_state_dict(
                    {key[7:]: dat['model'][key] for key in dat['model']})
            else:
                model.load_state_dict(dat['model'])
        elif 'state_dict' in dat:
            model.load_state_dict(dat['state_dict'])
        elif 'net' in dat:
            model.load_state_dict(dat['net'])
        elif 'main.0.weight' in dat:
            model.load_state_dict(dat)
        elif '_fc.weight' in dat:
            model.load_state_dict(dat)
        elif 'conv1.weight' in dat:
            model.load_state_dict(dat)
        else:
            print(list(dat.keys()))
            assert False
    return model
