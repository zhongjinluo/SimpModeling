import sys
import os
import json
import time
from PIL import Image
import numpy as np
from scipy.linalg import expm, norm
from numpy import cross, eye, dot
from flask import Flask, request
from sketch2norm import Sketch2Norm
from sketch2model import Sketch2Model
import trimesh
import requests
import _thread

app = Flask(__name__)

s2n = Sketch2Norm()
s2m = Sketch2Model()

def normalize(mesh_vertices):
    bbox_min = np.min(mesh_vertices, axis=0)
    bbox_max = np.max(mesh_vertices, axis=0)
    center = (bbox_min + bbox_max) / 2
    mesh_vertices -=  center
    r = np.max(np.sqrt(np.sum(np.array(mesh_vertices**2), axis=-1)))
    mesh_vertices /= r
    return mesh_vertices, center, r

def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))

@app.route('/msketch2norm', methods=["POST"])
def generate_norm():
    data = request.get_data()
    json_data = json.loads(data)
    sketch = np.array(json_data["front_sketch"], dtype=np.uint8).reshape(256, 256, 3)
    s, n = s2n.predict(sketch)
    print(s.shape)
    return {
        "front_sketch": s.reshape(-1).tolist(), 
        "front_norm": n.reshape(-1).tolist(),
        "side_sketch": s.reshape(-1).tolist(), 
        "side_norm": n.reshape(-1).tolist()
        }
        
def get_data(path, case_face=[1,1,1], case_ear=[1,1,0], case_horn=[1,1,0]):
    data = np.load(path)
    contour = []
    # face
    if case_face[0]:
        contour.append(data['face0'])
    if case_face[1]:
        contour.append(data['face1'])
    if case_face[2] and len(data["face2"]) > 1:
        contour.append(data['face2'])
    # ear
    if case_ear[0] and len(data["ear0"]) > 1:
        contour.append(data['ear0'])
    if case_ear[1] and len(data["ear1"]) > 1:
        contour.append(data['ear1'])
    if case_ear[2] and len(data["ear2"]) > 1:
        contour.append(data['ear2'])
    # horn
    if case_horn[0] and len(data["horn0"]) > 1:
        contour.append(data['horn0'])
    if case_horn[1] and len(data["horn1"]) > 1:
        contour.append(data['horn1'])
    if case_horn[2] and len(data["horn2"]) > 1:
        contour.append(data['horn2'])
    contour = np.concatenate(contour)
    return contour, data["vertices"], data["faces"]

def post(data):
    requests.post("http://10.26.2.35:5257/msketch2model", data=data)

@app.route('/msketch2model', methods=["POST"])
# @app.route('/msketch2model')
def generate_m():
    start = time.time()
    data = request.get_data()
    json_data = json.loads(data)
    # _thread.start_new_thread(post, (data, ))

    # save_dir = "results/"+json_data["name"]
    save_dir = "./results/"+json_data["name"].replace(" ", "-")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    front_sketch_512 = np.array(json_data["front_sketch_512"], dtype=np.uint8).reshape(512, 512, 3)
    Image.fromarray(np.uint8(front_sketch_512)).save(os.path.join(save_dir, "512.png"))

    total_s = time.time()
    front_sketch = np.array(json_data["front_sketch"], dtype=np.uint8).reshape(256, 256, 3)
    front_depth = np.array(json_data["side_sketch"], dtype=np.uint8).reshape(256, 256, 3) # side->depth
    back_depth = np.array(json_data["addition"], dtype=np.uint8).reshape(256, 256, 3)
    f_s, f_n = s2n.predict(front_sketch)
    end = time.time()
    # print("N:", end-start)

    vertices = np.array(json_data["vertices"], dtype=np.float64)
    faces = np.array(json_data["faces"], dtype=np.int)
    constraint = np.array(json_data["constraint"], dtype=np.int)
    constraint_faces = np.array(json_data["constraint_faces"], dtype=np.int)
    part_boundary = np.array(json_data["part_boundary"], dtype=np.int)
    inner_constraints = np.array(json_data["inner_constraints"], dtype=np.int)
    is_global = json_data["is_global"]
    
    vertices, c, r = normalize(vertices)
    M0 = M([0, 1, 0], np.pi / 2)
    vertices = dot(M0, vertices.T).T
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(os.path.join(save_dir, "before.obj"))
    start = time.time()
    vertices, faces = s2m.predict_with_template(f_s, f_n, front_depth, back_depth, vertices, faces, constraint, constraint_faces, part_boundary, inner_constraints, is_global, save_dir)
    M0 = M([0, 1, 0], -np.pi / 2)
    vertices = dot(M0, vertices.T).T
    vertices *= r
    vertices += c
    end = time.time()
    # print("M:", end-start)

    # total_e = time.time()
    # fo = open(os.path.join(save_dir, "256"), "w")
    # fo.write(str(total_e-total_s)+"\n")
    # fo.close()

    return {"vertices": vertices.tolist(), "faces": faces.tolist()}

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=8002)