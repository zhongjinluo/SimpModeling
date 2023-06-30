import data_processing.implicit_waterproofing as iw
import mcubes
import trimesh
import torch
import os
from glob import glob
import numpy as np
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from scipy.sparse import identity, csr_matrix

#import torch

class Generator(object):
    def __init__(self,
                 model,
                 threshold,
                 checkpoint_path,
                 device=torch.device("cuda"),
                 resolution=128,
                 batch_points=400000):
        self.model = model.to(device)
        #self.model = torch.nn.DataParallel(model).to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.checkpoint_path = checkpoint_path
        self.load_checkpoint(checkpoint_path)
        self.batch_points = batch_points

        self.min = -1
        self.max = 1

        grid_points = iw.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        grid_coords = grid_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = grid_points[:, 2], grid_points[:, 0]

        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)
    
    '''
    def get_new_vertices(self, V, faces, feature_indices, inputs, A, step=0.020):
        mesh = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=faces, process=False)
        vertex_normals = mesh.vertex_normals
        vertex_normals = vertex_normals / np.sqrt(np.sum(np.array(vertex_normals**2), axis=-1)).reshape(-1, 1)
        vertex_normals = torch.from_numpy(vertex_normals).float().cuda()
        # smooth normal
        vertex_normals = A.mm(vertex_normals)

        logits = self.model(V.reshape(-1, V.shape[0], V.shape[1]), inputs).squeeze(0)
        logits = torch.sigmoid(logits)
        initial_sign = torch.sign(logits - 0.5)
        count = 0
        while True:
            logits = self.model(V.reshape(-1, V.shape[0], V.shape[1]), inputs).squeeze(0)
            logits = torch.sigmoid(logits)
            p = logits -0.5
            V_bias = step * p.reshape(-1, 1) * vertex_normals
            V_tmp = V + V_bias

            sign = torch.sign(p)
            flags = sign - initial_sign
            change_indices = torch.eq(flags, 0)
            V[change_indices] = V_tmp[change_indices]

            check = torch.abs(logits-0.5)
            print(check[check<0.1].shape[0])
            if check[check<0.1].shape[0] > V.shape[0] * 0.90 or count > 10:
                break
            count += 1
            step *= 0.8
        return V
    
    def move(self, data):
        inputs = data['inputs'].to(self.device)
        vertices = data['vertices']
        faces = data["faces"]
        feature_indices = data["feature_indices"]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        A = trimesh.smoothing.laplacian_calculation(mesh)
        A = torch.from_numpy(A.toarray()).float().to(self.device)

        vertices = torch.from_numpy(vertices).float().to(self.device)
        with torch.no_grad():
            for i in range(2):
                print(i, vertices.shape)
                V_bar = self.get_new_vertices(vertices, faces, feature_indices, inputs, A, 0.050/(i+1))
                V_bar = A.mm(V_bar)
                vertices = V_bar

        return vertices.cpu().detach().numpy()
    '''

    def move(self, data):
        inputs = data['inputs'].to(self.device)
        vertices = data['vertices']
        faces = data["faces"]
        feature_indices = data["feature_indices"]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        A = trimesh.smoothing.laplacian_calculation(mesh)
        A = torch.from_numpy(A.toarray()).float().cuda()
        vertices = torch.from_numpy(vertices).float().cuda()
        with torch.no_grad():
            for i in range(6):
                V_bar = self.get_new_vertices(vertices, faces, feature_indices, inputs, A, step=0.025)
                V_bar = A.mm(V_bar)
                vertices = V_bar
        return vertices.cpu().detach().numpy()

    def get_new_vertices(self, vertices, faces, feature_indices, inputs, A, step=0.020):
        logits = self.model(vertices.reshape(-1, vertices.shape[0], vertices.shape[1]), inputs).squeeze(0)
        logits = torch.sigmoid(logits)
        sign = logits - 0.5 # torch.sign(logits - 0.5) 
        mesh = trimesh.Trimesh(vertices=vertices.cpu().detach().numpy(), faces=faces, process=False)
        vertex_normals = mesh.vertex_normals
        vertex_normals = vertex_normals / np.sqrt(np.sum(np.array(vertex_normals**2), axis=-1)).reshape(-1, 1)
        vertex_normals = torch.from_numpy(vertex_normals).float().cuda()
        # smooth normal
        vertex_normals = A.mm(vertex_normals)
        # smooth V_bias
        V_bias = step * sign.reshape(-1, 1) * vertex_normals
        V_bias[feature_indices] = torch.zeros((feature_indices.shape[0], 3)).cuda()
        V_bias = A.mm(V_bias)
        V_bar = vertices + V_bias
        return V_bar

    '''
    def get_new_vertices(self, vertices, faces, feature_indices, inputs, A, fix_weight, type=0):
        if type == 0:
            step = 0.02
            logits = self.model(vertices.reshape(-1, vertices.shape[0], vertices.shape[1]), inputs).squeeze(0)
            logits = torch.sigmoid(logits)
            sign = logits - 0.5 # torch.sign(logits - 0.5) 
            mesh = trimesh.Trimesh(vertices=vertices.cpu().detach().numpy(), faces=faces, process=False)
            vertex_normals = mesh.vertex_normals
            vertex_normals = vertex_normals / np.sqrt(np.sum(np.array(vertex_normals**2), axis=-1)).reshape(-1, 1)
            vertex_normals = torch.from_numpy(vertex_normals).float().to(self.device)
            # smooth normal
            vertex_normals = A.mm(vertex_normals)
            # smooth V_bias
            V_bias = step * sign.reshape(-1, 1) * vertex_normals
            V_bias[feature_indices] = torch.zeros((feature_indices.shape[0], 3)).to(self.device)
            V_bias = A.mm(V_bias)
            V_bar = vertices + V_bias
            V_bar[feature_indices] *= fix_weight
        else:
            pass
        return V_bar

    def move(self, data):
        inputs = data['inputs'].to(self.device)
        vertices = data['vertices']
        faces = data["faces"]
        feature_indices = data["feature_indices"]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        smooth_weight = 0.8
        target_weight = 1.0
        fix_weight = 1.0
        A = trimesh.smoothing.laplacian_calculation(mesh)
        I = identity(A.shape[0])
        A = torch.from_numpy(A.toarray()).float().to(self.device)
        I = torch.from_numpy(I.toarray()).float().to(self.device)
        L = (I - A)
        H = I.clone()
        D = torch.zeros((L.shape[0], 3)).to(self.device)
        H[feature_indices] *= fix_weight
        L_H = torch.cat([L * smooth_weight, H * target_weight])
        Q, R = torch.qr(L_H, some=True)
        R_inv = torch.inverse(R)
        Q_T = Q.t()

        vertices = torch.from_numpy(vertices).float().to(self.device)
        # anchors = vertices[feature_indices].clone()

        # local_weight = 0.7
        # anchor_weight = 1.0
        # H_2 = torch.zeros((feature_indices.shape[0], L.shape[1])).to(self.device)
        # for i in range(feature_indices.shape[0]):
        #     H_2[i, feature_indices[i]] = 1.0
        # L_H_2 = torch.cat([L * local_weight, H_2 * anchor_weight])
        # V_bar_2 = anchors
        # Q_2, R_2 = torch.qr(L_H_2, some=True)
        # R_inv_2 = torch.inverse(R_2)
        # Q_T_2 = Q_2.t()

        with torch.no_grad():
            for i in range(6):
                V_bar = self.get_new_vertices(vertices, faces, feature_indices, inputs, A, fix_weight)
                D_V_bar = torch.cat([D * smooth_weight, V_bar * target_weight])
                vertices = R_inv @ Q_T @ D_V_bar

                # D_2 = L.mm(vertices)
                # D_V_bar_2 = torch.cat([D_2 * local_weight , V_bar_2 * anchor_weight])
                # vertices = R_inv_2 @ Q_T_2 @ D_V_bar_2

        return vertices.cpu().detach().numpy()
    '''

    def generate_mesh(self, data):

        inputs = data['inputs'].to(self.device)

        logits_list = []
        for points in self.grid_points_split:
            with torch.no_grad():
                logits = self.model(points, inputs)
                #logits = torch.sigmoid(logits)
            logits_list.append(logits.squeeze(0).detach().cpu())

        logits = torch.cat(logits_list, dim=0)

        return logits.numpy()

    def mesh_from_logits(self, logits):
        logits = np.reshape(logits, (self.resolution,) * 3)

        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        vertices, triangles = mcubes.marching_cubes(logits, threshold)
        #vertices, triangles = mcubes.marching_cubes(logits, 0.5)

        # remove translation due to padding
        vertices -= 1

        # rescale to original scale
        step = (self.max - self.min) / (self.resolution - 1)
        vertices = np.multiply(vertices, step)
        vertices += [self.min, self.min, self.min]

        return trimesh.Trimesh(vertices, triangles)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
