
import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from normal.options.test_options import TestOptions
# from data.data_loader import CreateDataLoader
from normal.models.models import create_model
import normal.util.util as util
from normal.util import html
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class Sketch2Norm:
    def __init__(self):
        self.opt = TestOptions().parse(save=False)
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.name = "GapMesh"
        self.opt.checkpoints_dir = "normal/checkpoints"
        self.opt.label_nc = 0
        self.opt.dataroot = "../../datasets/IF-PIFU-MINI-BACK"
        self.opt.loadSize = 256
        self.opt.no_instance = True
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.model = create_model(self.opt).cuda()
        
    def predict(self, image):
        with torch.no_grad():
            input_img = self.to_tensor(image).unsqueeze(0).cuda()
            generated = self.model.inference(input_img, None, None)
            print(input_img.shape)
            norm = util.tensor2im(generated.data[0])
            return image, norm

if __name__ == '__main__':
    pass


