import sys
import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.lib.options import BaseOptions
from model.lib.mesh_util import *
from model.lib.sample_util import *
from model.lib.train_util import *
from model.lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

class Sketch2Model:
    def __init__(self, projection_mode='orthogonal'):
        # get options
        opt = BaseOptions().parse()
        opt.loadSize = 256
        self.opt = opt

        self.load_size = opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNetwNML(opt, projection_mode).cuda()
        netG.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), "model/checkpoints/example/netG_latest"), map_location=cuda))

        self.cuda = cuda
        self.netG = netG

        with open("model/PV.json", 'r') as f:
            data = json.load(f)
            P = np.array(data["P"]).reshape(-1, 4)
            P[1, 1] = -P[1, 1]
            V = np.array(data["V"]).reshape(-1, 4)
            self.calib = torch.from_numpy(P.dot(V)).float()

    def predict(self, sketch, norm, depth, depth_back_plus, save_dir):
        sketch = self.to_tensor(Image.fromarray(sketch).convert('RGB'))
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        depth = self.to_tensor(Image.fromarray(depth).convert('RGB'))
        depth_back_plus = self.to_tensor(Image.fromarray(depth_back_plus).convert('RGB'))
        render = torch.cat([sketch, norm, depth, depth_back_plus], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            self.netG.eval()
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = self.calib.unsqueeze(0)
            return gen_mesh(self.opt, self.netG, self.cuda, data, os.path.join(save_dir, "pifu.obj"))
    
    def predict_with_template(self, sketch, norm, depth, depth_back_plus, vertices, faces, constraint, constraint_faces, part_boundary, inner_constraints, is_global, save_dir):
        sketch = self.to_tensor(Image.fromarray(sketch).convert('RGB'))
        norm = self.to_tensor(Image.fromarray(norm).convert('RGB'))
        depth = self.to_tensor(Image.fromarray(depth).convert('RGB'))
        depth_back_plus = self.to_tensor(Image.fromarray(depth_back_plus).convert('RGB'))
        render = torch.cat([sketch, norm, depth, depth_back_plus], 0).unsqueeze(0)
        data = {}
        with torch.no_grad():
            self.netG.eval()
            data['img'] = render
            data['name'] = "current"
            data['b_min'] = np.array([-1, -1, -1])
            data['b_max'] = np.array([1, 1, 1])
            data['calib'] = self.calib.unsqueeze(0)
            # gen_mesh(self.opt, self.netG, self.cuda, data, os.path.join(save_dir, "pifu.obj"))
            return gen_mesh_with_template(self.opt, self.netG, self.cuda, data, os.path.join(save_dir, "pifu.obj"), vertices=vertices, faces=faces, constraint=constraint, constraint_faces=constraint_faces, part_boundary=part_boundary, inner_constraints=inner_constraints, is_global=is_global)

if __name__ == '__main__':
    pass