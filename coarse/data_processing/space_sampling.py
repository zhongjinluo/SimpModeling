import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback

ROOT = 'animals/data'


def boundary_sampling(path):
    try:
        off_path = path + '/mesh.off'
        out_file = path +'/space_samples.npz'

        mesh = trimesh.load(off_path)

        grid_points = iw.create_grid_points_from_bounds(-1, 1, 64)
        occupancies = iw.implicit_waterproofing(mesh, grid_points)[0]
        
        grid_coords = grid_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = grid_points[:, 2], grid_points[:, 0]

        np.savez(out_file, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run space sampling'
    )
    parser.add_argument('-sigma', type=float)

    args = parser.parse_args()

    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, glob.glob( ROOT + '/*/'))
