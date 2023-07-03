from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import scipy.io as sio
import json

log = logging.getLogger('trimesh')
log.setLevel(40)

def crop_image(img, rect):
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0
    
    if img.shape[2] == 4:
        color = [0, 0, 0, 0]
    else:
        color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top

    return new_img[y:(y+h),x:(x+w),:]

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        if (f[-4:] == '.obj'):
            meshs[f[:-4]] = trimesh.load(os.path.join(root_dir, f))
    return meshs

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


class TrainDataset_Refine(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.OBJ = os.path.join(self.root, 'GEO')
        self.CALIB = os.path.join(self.root, 'CALIB')
        self.NORM = os.path.join(self.root, 'NORM')

        self.B_MIN = np.array([-1, -1, -1])
        self.B_MAX = np.array([1, 1, 1])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.subjects = self.get_subjects()
        self.augs = 1

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
        self.mesh_dic = load_trimesh(self.OBJ)

    def get_img_info(self, subject):
        calib_list = []
        render_list = []

        render_path = os.path.join(self.RENDER, subject+'.png')
        calib_path = os.path.join(self.CALIB, "PV.json")
        fnorm_path = os.path.join(self.NORM, subject+'.png')

        render = Image.open(render_path).convert('RGB')
        fnorm = Image.open(fnorm_path).convert('RGB')
        render = render.resize((512, 512))
        fnorm = fnorm.resize((512, 512))

        with open(calib_path, 'r') as f:
            data = json.load(f)
            P = np.array(data["P"]).reshape(-1, 4)
            P[1, 1] = -P[1, 1]
            V = np.array(data["V"]).reshape(-1, 4)
            calib = torch.from_numpy(P.dot(V)).float()

        if self.is_train:
            render = self.aug_trans(render)
            if self.opt.aug_blur > 0.00001:
                blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                render = render.filter(blur)

        render = self.to_tensor(render)
        fnorm = self.to_tensor(fnorm)

        render = torch.cat([render, fnorm], 0)

        render_list.append(render)
        calib_list.append(calib)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
        }
    def get_subjects(self):
        all_subjects = []
        files = os.listdir(self.RENDER)
        for file in files:
                all_subjects.append(file[:-4])
        return sorted(list(all_subjects))[0:900]

    def __len__(self):
        return len(self.subjects) * self.augs

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        # print(subject)
        mesh = self.mesh_dic[subject]
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma*2, size=surface_points.shape)

        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]
        samples = np.concatenate([inside_points, outside_points], 0).T
        # import openmesh as om
        # c = om.TriMesh(points=samples.T)
        # om.write_mesh("sample.obj", c)
        # print(samples.shape)

        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        # save_samples_truncted_prob('out.ply', samples.T, labels.T)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()

        del mesh

        return {
             'samples': samples,
             'labels': labels
        }


    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index // self.augs

        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_img_info(subject)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)

        return res

    def __getitem__(self, index):
        return self.get_item(index)
