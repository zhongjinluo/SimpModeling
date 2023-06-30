from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
import trimesh
import openmesh as om
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from scipy.sparse import identity, csr_matrix
import time

def reconstructionLM(net, cuda, calib_tensor, target,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=1000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    locX = int(target[0])
    locY = int(target[1])
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid_LM(locX, locY, resolution,
                              b_min, b_max, transform=transform)
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]

        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    print('=== target: ', target)
    print('=== coords: ', coords.shape)
    print('=== sdf: ', np.max(sdf), np.min(sdf), np.mean(sdf))
    ind = np.unravel_index(np.argmax(sdf, axis=None), sdf.shape)
    print('=== lm[0] should at: ', coords[:,ind[0], ind[1], ind[2]])
    return ind, sdf
    

def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, num_samples=10000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution,
                              b_min, b_max, transform=transform)
    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, net.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]

        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)
    
    #print('=== sdf: ', sdf.shape, ' ===')
    #print(sdf)
    # Finally we do marching cubes
    try:
        # print(sdf)
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5)
        # transform verts into world coordinate system
        verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
        verts = verts.T
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1

# def get_new_vertices(net, cuda, calib_tensor, V, faces, A = None, step=0.010, batch_size=80000):
def get_new_vertices(net, cuda, calib_tensor, V, faces, A = None, step=0.010, batch_size=200000):
    mesh = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals = mesh.vertex_normals
    vertex_normals = vertex_normals / np.sqrt(np.sum(np.array(vertex_normals**2), axis=-1)).reshape(-1, 1)
    vertex_normals = torch.from_numpy(vertex_normals).float().cuda()
    # smooth normal
    # vertex_normals = A.mm(vertex_normals)
    steps = batch_size // V.shape[0]
    bias = torch.linspace(0.0, 0.06, steps=steps).cuda()
    bias = bias.reshape(-1, 1).repeat(1, V.shape[0])
    bias = bias.reshape(-1, V.shape[0], 1)
    net.query(V.T.unsqueeze(0), calib_tensor)
    logits = net.get_preds()[0][0]
    initial_sign = torch.sign(logits - 0.5)
    directions = initial_sign.reshape(-1, 1) * vertex_normals
    V_sample =  V.unsqueeze(0).repeat(bias.shape[0], 1, 1) + bias * directions
    V_sample = V_sample.reshape(-1, 3)
    net.query(V_sample.T.unsqueeze(0), calib_tensor)
    logits = net.get_preds()[0][0]
    sign = torch.sign(logits - 0.5).reshape(bias.shape[0], -1, 1)
    sign = sign - sign[0, :, :]
    sign = sign.squeeze(-1).T
    sign = torch.ne(sign, 0) + 0
    idx =  torch.arange(sign.shape[1], 0, -1).cuda()
    sign= sign * idx
    indices_2 = torch.argmax(sign, 1, keepdim=True).reshape(-1)  # torch.argmax(sign, 1, keepdim=True).reshape(-1) - 1
    indices_1 = torch.arange(0, sign.shape[0], 1).cuda()
    V_sample =  V_sample.reshape(bias.shape[0], -1, 3)
    V_new = V_sample[indices_2, indices_1, :]
    return V_new

def reconstruction_with_template(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, vertices=None, faces=None, constraint=None, constraint_faces=None, part_boundary=None, inner_constraints=None, num_samples=10000, transform=None):

    start = time.time()
    j = 0
    v_map = {}
    v_map_r = {}
    part_vertices = []
    for i in constraint:
        part_vertices.append(vertices[i])
        v_map[i] = j
        v_map_r[j] = i
        j += 1
    part_faces = []
    for i in constraint_faces:
        f = faces[i]
        part_faces.append([v_map[f[0]], v_map[f[1]], v_map[f[2]]])
    mesh = trimesh.Trimesh(vertices=part_vertices, faces=part_faces, process=False)
    # mesh.export("checks/part_raw.obj")

    outter = []
    outter_indices = []
    for i in part_boundary:
        outter.append(part_vertices[v_map[i]])
        outter_indices.append(v_map[i])
    c = om.TriMesh(points=outter)
    # om.write_mesh("checks/outter.obj", c)
    inner = []
    inner_indices = []
    for i in inner_constraints:
        inner.append(part_vertices[v_map[i]])
        inner_indices.append(v_map[i])
    c = om.TriMesh(points=inner)
    # om.write_mesh("checks/inner.obj", c)
    end = time.time()
    print("part:", end-start)

    l_weight = 0.5
    target_weight = 1.0
    fix__weight = 1.2
    A = trimesh.smoothing.laplacian_calculation(mesh)
    A = torch.from_numpy(A.toarray()).float().cuda()
    L = torch.from_numpy(identity(A.shape[0]).toarray()).float().cuda() - A
    D = torch.zeros((L.shape[0], 3)).cuda()
    H_outter = torch.zeros((len(outter_indices), L.shape[1])).cuda()
    H_inner =  torch.zeros((len(inner_indices), L.shape[1])).cuda()
    for i in range(len(outter_indices)):
        H_outter[i, outter_indices[i]] = 1.0
    for i in range(len(inner_indices)):
        H_inner[i, inner_indices[i]] = 1.0

    L_H = torch.cat([L * l_weight, H_outter * target_weight, H_inner * target_weight])

    start = time.time()
    V = torch.from_numpy(mesh.vertices).float().cuda()
    D = L.mm(V)
    with torch.no_grad():
        for i in range(1):
            V_bar = get_new_vertices(net, cuda, calib_tensor, V, mesh.faces, A=A, step = 0.010 / (i+1))
            V_outter = V[outter_indices]
            V_inner = V_bar[inner_indices]
            D_V_bar = torch.cat([D * l_weight, V_outter * target_weight, V_inner * target_weight])
            V, _ = torch.lstsq(D_V_bar, L_H)
            V = V[0:D.shape[0]]
            mask = torch.ones(V.shape[0], dtype=bool)
            mask[outter_indices] = False
            V[mask] = A.mm(V)[mask]
    vertices[constraint] = V.detach().cpu().numpy()
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # A = trimesh.smoothing.laplacian_calculation(mesh)
    # A = torch.from_numpy(A.toarray()).float().cuda()
    # V = torch.from_numpy(mesh.vertices).float().cuda()
    # V = A.mm(V)
    # vertices = V.detach().cpu().numpy()
    end = time.time()
    print("get_new_vertices:", end-start)
    

    return vertices, faces


# 全局变形
def get_new_vertices_global(net, cuda, calib_tensor, V, faces, A = None, step=0.010, batch_size=300000):
    mesh = trimesh.Trimesh(vertices=V.detach().cpu().numpy(), faces=faces, process=False)
    vertex_normals = mesh.vertex_normals
    vertex_normals = vertex_normals / np.sqrt(np.sum(np.array(vertex_normals**2), axis=-1)).reshape(-1, 1)
    vertex_normals = torch.from_numpy(vertex_normals).float().cuda()
    # smooth normal
    # vertex_normals = A.mm(vertex_normals)
    steps = batch_size // V.shape[0]
    bias = torch.linspace(0.0, step, steps=steps).cuda()
    bias = bias.reshape(-1, 1).repeat(1, V.shape[0])
    bias = bias.reshape(-1, V.shape[0], 1)
    net.query(V.T.unsqueeze(0), calib_tensor)
    logits = net.get_preds()[0][0]
    initial_sign = torch.sign(logits - 0.5)
    directions = initial_sign.reshape(-1, 1) * vertex_normals
    V_sample =  V.unsqueeze(0).repeat(bias.shape[0], 1, 1) + bias * directions
    V_sample = V_sample.reshape(-1, 3)
    net.query(V_sample.T.unsqueeze(0), calib_tensor)
    logits = net.get_preds()[0][0]
    sign = torch.sign(logits - 0.5).reshape(bias.shape[0], -1, 1)
    sign = sign - sign[0, :, :]
    sign = sign.squeeze(-1).T
    sign = torch.ne(sign, 0) + 0
    idx =  torch.arange(sign.shape[1], 0, -1).cuda()
    sign= sign * idx
    indices_2 = torch.argmax(sign, 1, keepdim=True).reshape(-1)  # torch.argmax(sign, 1, keepdim=True).reshape(-1) - 1
    indices_1 = torch.arange(0, sign.shape[0], 1).cuda()
    V_sample =  V_sample.reshape(bias.shape[0], -1, 3)
    V_new = V_sample[indices_2, indices_1, :]
    return V_new

def reconstruction_with_template_global(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, vertices=None, faces=None, constraint=None, constraint_faces=None, part_boundary=None, inner_constraints=None, num_samples=80000, transform=None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    A = trimesh.smoothing.laplacian_calculation(mesh)
    A = torch.from_numpy(A.toarray()).float().cuda()
    V = torch.from_numpy(vertices).float().cuda()
    with torch.no_grad():
        for i in range(6):
            V_bar = get_new_vertices_global(net, cuda, calib_tensor, V, mesh.faces, A=A, step = 0.060/(i+1))
            # V_bar = A.mm(V_bar)
            V = V_bar
    vertices = V.detach().cpu().numpy()
    return vertices, faces
'''
def reconstruction_with_template_global(net, cuda, calib_tensor,
                   resolution, b_min, b_max,
                   use_octree=False, vertices=None, faces=None, constraint=None, constraint_faces=None, part_boundary=None, inner_constraints=None, num_samples=80000, transform=None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    smooth_weight = 0.66
    target_weight = 1.0
    A = trimesh.smoothing.laplacian_calculation(mesh)
    I = identity(A.shape[0])
    A = torch.from_numpy(A.toarray()).float().cuda()
    I = torch.from_numpy(I.toarray()).float().cuda()
    L = (I - A)
    H = I.clone()
    # D = torch.zeros((L.shape[0], 3)).cuda()
    L_H = torch.cat([L * smooth_weight, H * target_weight])
    
    start = time.time()
    V = torch.from_numpy(mesh.vertices).float().cuda()
    D = L.mm(V)
    with torch.no_grad():
        for i in range(1):
            V_bar = get_new_vertices_global(net, cuda, calib_tensor, V, mesh.faces, A=A, step = 0.020/(i+1))
            D_V_bar = torch.cat([D * smooth_weight, V_bar * target_weight])
            V, _ = torch.lstsq(D_V_bar, L_H)
    vertices = V.detach().cpu().numpy()[0:D.shape[0]]
    end = time.time()
    print("deformation:", end-start)
    return vertices, faces
'''

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()
    # print("OK---------------------")


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
