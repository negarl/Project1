import numpy as np
import os
import pywavefront
from mask_pointcloud_joint import object_point_cloud


def read_model_object_points(object_name, models_folder_path):
    """

    :param object_name:
    :param models_folder_path:
    :return:
    """
    folders_name = [folder for folder in os.listdir(models_folder_path) if os.path.isdir(os.path.join(models_folder_path, folder))]
    folder_name = [folder for folder in folders_name if folder.find(object_name) != -1][0]
    object_source_cloud = pywavefront.Wavefront(os.path.join(models_folder_path,folder_name,"textured_simple.obj"))
    return np.asarray(object_source_cloud.vertices)


def get_objects_point_file(scene_folder_path, objects_file_names, pcd_file_name):
    """

    :param scene_folder_path:
    :param objects_file_names:
    :param pcd_file_name:
    :return:
    """

    objects_points_dic = {}
    objects_colors_dic = {}

    for object_name in objects_file_names:
        name_separated = (object_name.split(".")[0]).split("_")[:-1]
        name = '_'.join([str(elem) for elem in name_separated])
        objects_points_dic.setdefault(name)
        objects_colors_dic.setdefault(name)
        points_array, colors_array = object_point_cloud(os.path.join(scene_folder_path, object_name), os.path.join(scene_folder_path, pcd_file_name))
        objects_points_dic[name] = points_array
        objects_colors_dic[name] = colors_array

    return objects_points_dic, objects_colors_dic


def read_scene_objects_point_files(scene_folder_path):
    """

    :param scene_folder_path:
    :return:
    """
    files_names = [file_name for file_name in os.listdir(scene_folder_path) if os.path.isfile(os.path.join(scene_folder_path, file_name)) and file_name[0] != "."]
    pcd_file_name = [file_name for file_name in files_names if file_name.split(".")[-1] == "pcd"][0]
    objects_files_names = [file_name for file_name in files_names if file_name.split(".")[-1] == "png"]
    objects_points_dic, objects_colors_dic = get_objects_point_file(scene_folder_path, objects_files_names, pcd_file_name)
    return objects_points_dic, objects_colors_dic




if __name__ == '__main__':
    c = "/home/user/Documents/Negar/TEASER Project/models"
    b = "/home/user/Documents/Negar/TEASER Project/teaser_scene/scene3"
    read_model_object_points("sugar_box", c)
    #read_scene_objects_point_files(b)