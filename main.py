import os
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mask_pointcloud_joint import point_cloud_array_to_image
from read_file import read_scene_objects_point_files, read_model_object_points, read_pose_file
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D

# Import TEASER++
sys.path.append('/home/user/TEASER/TEASER-plusplus/build/python/teaserpp_python');
import teaserpp_python

NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10


def calculate_objects_rotation(dst, src, solver):

    print("dim: ", src.shape, dst.shape)

    idx_s = random.sample(range(src.shape[0]), 100)
    s = src[idx_s, :]
    idx_d = random.sample(range(dst.shape[0]), 100)
    d = dst[idx_d, :]

    new_src = np.tile(s, (d.shape[0], 1))
    new_dest = np.tile(d, (s.shape[0], 1))

    new_dest = np.moveaxis(new_dest, -1, 0)
    new_src = np.moveaxis(new_src, -1, 0)
    print("dim: ", new_src.shape, new_dest.shape)

    solver.solve(new_src, new_dest)

    solution = solver.getSolution()

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")

    print("Estimated rotation: ")
    estimated_rotation = solution.rotation
    print(estimated_rotation)

    print("Estimated translation: ")
    estimated_translation = solution.translation
    print(estimated_translation)

    return estimated_translation, estimated_rotation


def plot_object_rotations(rotated_src_array, dst_array, object_name, scene_name):

    xd = dst_array[:, 0]
    yd = dst_array[:, 1]
    zd = dst_array[:, 2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xd, yd, zd)

    xr = rotated_src_array[:, 0]
    yr = rotated_src_array[:, 1]
    zr = rotated_src_array[:, 2]
    ax.scatter(xr, yr, zr)

    file_name = object_name + ".png"
    folder_path = "/home/user/PycharmProjects/TEASERPP/object_rotations/" + scene_name
    plt.savefig(os.path.join(folder_path, file_name))
    #plt.show()

    return


def convert_quaternion_to_rotation_matrix(rotation_q_array):
    q_array = R.from_quat(rotation_q_array)
    rotation_matrix = q_array.as_matrix()
    return rotation_matrix


def convert_rotation_matrix_to_quaternion(rotation_matrix):
    q_array = R.from_matrix(rotation_matrix)
    rotation_q_array = q_array.as_quat()
    return rotation_q_array


def calculate_object_rotation_difference(real_q_rotation, estimated_rotation_matrix):

    estimated_q_rotation = convert_rotation_matrix_to_quaternion(estimated_rotation_matrix)
    difference = real_q_rotation - estimated_q_rotation
    return difference, estimated_q_rotation


def measure_object_predictions(dst, src, solver, object_real_translation, object_real_q_rotation, object_name, scene_name):

    estimated_translation, estimated_rotation = calculate_objects_rotation(dst, src, solver)
    after_rotate = src.dot(estimated_rotation)
    rotated_src_array = after_rotate + estimated_translation
    plot_object_rotations(rotated_src_array, dst, object_name, scene_name)
    object_rotation_difference, estimated_q_rotation = calculate_object_rotation_difference(object_real_q_rotation, estimated_rotation)
    object_translation_difference = object_real_translation - estimated_translation

    return np.asarray(object_translation_difference), np.asarray(object_rotation_difference), np.asarray(estimated_q_rotation)


def scene_objects_rotation(scene_folder_path, scene_name, models_folder_path, solver, iteration_number):

    objects_points_dic, objects_colors_dic = read_scene_objects_point_files(scene_folder_path)
    translation_dic, rotation_q_dic = read_pose_file(scene_folder_path)

    f = "file"

    if iteration_number == 1:
        file_name = scene_name + ".txt"
        f = open(file_name, "w")

    for object_name in objects_points_dic.keys():
        print(object_name)
        dst = objects_points_dic[object_name]
        dst_color = objects_colors_dic[object_name]
        point_cloud_array_to_image(dst, dst_color, object_name, scene_name)
        src = read_model_object_points(object_name, models_folder_path)
        src = src[:, :3]
        object_translation = translation_dic[object_name]
        object_q_rotation = rotation_q_dic[object_name]

        if iteration_number == 1 :
            object_translation_difference, object_rotation_difference, object_estimated_q_rotation = measure_object_predictions(dst, src, solver, object_translation, object_q_rotation, object_name, scene_name)
            f.write(object_name + "\n")
            f.write("estimated_q_rotation/" + str(object_estimated_q_rotation)+ "\n")
            f.write("translation_difference/" + str(object_translation_difference)+ "\n")
            f.write("rotation_difference/" + str(object_rotation_difference)+ "\n")
        else :

            avg_object_translation_difference = []
            avg_object_rotation_difference = []

            file_name = object_name + ".txt"
            file_path = "/home/user/PycharmProjects/TEASERPP/result_files/" + scene_name
            f = open(os.path.join(file_path,file_name), "w")

            for i in range(iteration_number):
                object_translation_difference, object_rotation_difference, object_estimated_q_rotation = measure_object_predictions(
                    dst, src, solver, object_translation, object_q_rotation, object_name, scene_name)

                if i == 0 :
                    avg_object_translation_difference = np.zeros(np.asarray(object_translation_difference).shape)
                    avg_object_rotation_difference = np.zeros(np.asarray(object_rotation_difference).shape)

                avg_object_translation_difference = avg_object_translation_difference + object_translation_difference
                avg_object_rotation_difference = avg_object_rotation_difference + object_rotation_difference

                f.write("iteration_number/ " + str(iteration_number) + "\n")
                f.write("estimated_q_rotation/ " + str(object_estimated_q_rotation) + "\n")
                f.write("translation_difference/ " + str(object_translation_difference) + "\n")
                f.write("rotation_difference/ " + str(object_rotation_difference) + "\n")

            avg_object_translation_difference = avg_object_translation_difference / iteration_number
            avg_object_rotation_difference = avg_object_rotation_difference / iteration_number

            f.write("avg_translation_difference/ " + str(avg_object_translation_difference) + "\n")
            f.write("avg_rotation_difference/ " + str(avg_object_rotation_difference) + "\n")
            f.close()

        print("ldgh ")
    if iteration_number == 1 :
        f.close()

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
    iteration_number = 1
    scene_objects_rotation(scene_folder_path, "scene3", models_folder_path, solver, iteration_number)
