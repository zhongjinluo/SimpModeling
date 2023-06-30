import argparse
import numpy as np
import data_processing.implicit_waterproofing as iw
from scipy.spatial import cKDTree as KDTree
from models.generation import Generator
import models.local_model as model
import trimesh
import torch
import openmesh as om
import os
import time
import json
from scipy.linalg import expm, norm
from numpy import cross, eye, dot
from flask import Flask, request
from plyfile import PlyData


def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    r *= 1.00
    mesh_vertices /= r
    return mesh_vertices, center, r

def parse_args():
    parser = argparse.ArgumentParser(description='Run prediction')
    parser.add_argument('-res' , default=128, type=int)
    parser.add_argument('-mode' , default='test', type=str)
    parser.add_argument('-retrieval_res' , default=128, type=int)
    parser.add_argument('-pc_samples', default=3000, type=int)
    parser.add_argument('-batch_points', default=4000, type=int)
    parser.add_argument('-checkpoint' , default='experiments/exp_3000v128/checkpoints/checkpoint_epoch_200.tar', type=str)
    args = parser.parse_args()

    return args

def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))

app = Flask(__name__)
args = parse_args()

net = model.AnimalsContour()
gen = Generator(net, 0.5, checkpoint_path=args.checkpoint ,resolution=args.retrieval_res, batch_points=args.batch_points)

grid_points = iw.create_grid_points_from_bounds(-1, 1, args.res)
occ = np.zeros(len(grid_points), dtype=np.int8)
kdtree = KDTree(grid_points)

def get_data(contour, kdtree, occ, num_samples, res):
    indices = np.random.randint(0, len(contour), num_samples)
    point_cloud = contour[indices]
    _, idx = kdtree.query(point_cloud)
    occupancies = occ.copy()
    occupancies[idx] = 1
    inputs = np.reshape(occupancies, (res,)*3)
    inputs = torch.from_numpy(np.array(inputs, dtype=np.float32)).unsqueeze(0)
    data = {'inputs': inputs, 'point_cloud':point_cloud}
    return data

def move(data):
    vertices = gen.move(data)
    return vertices

@app.route('/generate', methods=["POST"])
def generate():
    base_name = str(time.time())
    data = request.get_data()
    json_data = json.loads(data)
    
    vertices = np.array(json_data["vertices"], dtype=np.float64)
    faces = np.array(json_data["faces"], dtype=np.int)
    feature_indices = np.array(json_data["feature_indices"], dtype=np.int)
    feature_points = np.array(json_data["feature_points"], dtype=np.float64)

    vertices, c, r = normalize(vertices)
    contour = (feature_points-c)*(1/r)
    
    M0 = M([0, 1, 0], np.pi / 2)
    contour = dot(M0, contour.T).T

    data = get_data(contour, kdtree, occ, args.pc_samples, args.res)

    M0 = M([0, 1, 0], np.pi)
    vertices = dot(M0, vertices.T).T
    data["vertices"] = vertices
    data["faces"] = faces
    data["feature_indices"] = feature_indices
    vertices = move(data)
    M0 = M([0, 1, 0], -np.pi/2)
    vertices = dot(M0, vertices.T).T
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # mesh.export('../results/ifnet_our.obj')
    M0 = M([0, 1, 0], -np.pi/2)
    vertices = dot(M0, vertices.T).T
    vertices *= r
    vertices += c

    return {"vertices": vertices.tolist(), "faces": faces.tolist()}


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8001)
