import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from read_file import read_scene_objects_point_files, read_model_object_points
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D

# Import TEASER++
sys.path.append('/home/user/TEASER/TEASER-plusplus/build/python/teaserpp_python');
import teaserpp_python

NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10


def set_src_dimension(src):
    src = np.moveaxis(src, -1, 0)
    new_src = np.zeros((3, src.shape[1]))
    for i in range(src.shape[0]):
        if i == 3:
            break
        for j in range(src.shape[1]):
            new_src[i][j] = src[i][j]
    new_src = np.moveaxis(new_src, -1, 0)
    return new_src


def objects_rotation (dst, dst_color, src, solver):

    print("dim: ", src.shape, dst.shape)
    s = src[5000:5100, :]
    d = dst[5000:5100, :]

    new_src = np.tile(s, (d.shape[0], 1))
    new_dest = np.tile(d, (s.shape[0], 1))

    new_dest = np.moveaxis(new_dest, -1, 0)
    new_src = np.moveaxis(new_src, -1, 0)
    print("dim: ", new_src.shape, new_dest.shape)

    xd = dst[:, 0]
    yd = dst[:, 1]
    zd = dst[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xd, yd, zd)

    solver.solve(new_src, new_dest)

    solution = solver.getSolution()

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")

    print("Estimated rotation: ")
    print(solution.rotation)

    print("Estimated translation: ")
    print(solution.translation)

    after_rotate = src.dot(solution.rotation)
    after_rotate = after_rotate + solution.translation
    xr = after_rotate[:, 0]
    yr = after_rotate[:, 1]
    zr = after_rotate[:, 2]

    ax.scatter(xr, yr, zr)
    plt.show()
    return


def scene_objects_rotation(scene_folder_path, models_folder_path, solver):

    objects_points_dic, objects_colors_dic = read_scene_objects_point_files(scene_folder_path)

    for object_name in objects_points_dic.keys():
        print(object_name)
        dst = objects_points_dic[object_name]
        dst_color = objects_colors_dic[object_name]
        src = read_model_object_points(object_name, models_folder_path)
        src = set_src_dimension(src)
        objects_rotation(dst, dst_color, src, solver)

    return


if __name__ == "__main__":

    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    models_folder_path = "/home/user/Documents/Negar/TEASER Project/models"
    scene_folder_path = "/home/user/Documents/Negar/TEASER Project/teaser_scene/scene3"
    scene_objects_rotation(scene_folder_path, models_folder_path, solver)
