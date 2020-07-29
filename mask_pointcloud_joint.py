import math
import os
import cv2
import numpy as np
import open3d as o3d

"""
To find pint cloud of an object (Because our scene point cloud are not organized):]
    1. Convert point clouds to image (Function point_cloud_to_image)
    2. Find intersection of scene image and the point cloud image and set object point cloud (Function object_point_cloud)  
    3.
    4. 
"""


def camera_intrinsic_parameter():
    """
    Returning the intrinsic parameters of the pinhole camera (cx, cy, fx, fy)
    :return: return fx, fy, cx, cy respectively
    """

    """
    Camera intrinsic matrix shape : 
    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    camera_intrinsic_param = (o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)).intrinsic_matrix
    width = (o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)).width
    height = (o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)).height
    return width, height, camera_intrinsic_param[0][0], camera_intrinsic_param[1][1], camera_intrinsic_param[0][2], camera_intrinsic_param[1][2]


def point_cloud_to_image(scene_point_cloud_file_path):
    """
    Convert scene point clouds to image of the scene and save the image with name "point_cloud_image.jpg"
    :param scene_point_cloud_file_path: Path to the scene point cloud file
    :return: Scene image object
    """

    # Camera intrinsics  parameters
    width, height, fx, fy, cx, cy = camera_intrinsic_parameter()

    # Read the scene point cloud file
    scene_cloud_object = o3d.io.read_point_cloud(scene_point_cloud_file_path)
    # Scene point cloud array
    scene_point = (np.asarray(scene_cloud_object.points))
    # Scene RGB color array
    scene_colour = np.asarray(scene_cloud_object.colors)

    # Create an numpy array with the sane height and width of the scene image with initial value of 255
    point_cloud_image = np.full(shape=[height, width, 3], fill_value=255, dtype=float)

    # For each point, calculate the pixel location and set the RGB values
    for idx, pt in enumerate(scene_point):
        u = int(((pt[0] * fx) / pt[2]) + cx)
        v = int(((pt[1] * fy) / pt[2]) + cy)

        RGB_color = (scene_colour[idx] * 255)
        point_cloud_image[v][u] = RGB_color

    cv2.imwrite("point_cloud_image.jpg", point_cloud_image)

    return point_cloud_image


def point_cloud_array_to_image(scene_point, scene_colour, object_name, scene_name):
    """
    Convert scene point clouds to image of the scene and save the image with name "point_cloud_image.jpg"
    :param scene_point: Scene point clouds array
    :param scene_colour: Scene point color array
    :return: Scene image object
    """

    # Camera intrinsics  parameters
    width, height, fx, fy, cx, cy = camera_intrinsic_parameter()

    # Create an numpy array with the sane height and width of the scene image with initial value of 255
    point_cloud_image = np.full(shape=[height, width, 3], fill_value=255, dtype=float)

    # For each point, calculate the pixel location and set the RGB values
    for idx, pt in enumerate(scene_point):
        u = int(((pt[0] * fx) / pt[2]) + cx)
        v = int(((pt[1] * fy) / pt[2]) + cy)

        RGB_color = (scene_colour[idx] * 255)
        point_cloud_image[v][u] = RGB_color

    file_name = object_name + ".jpg"
    folder_path = "/home/user/PycharmProjects/TEASERPP/mask_point_picture/" + scene_name
    cv2.imwrite(os.path.join(folder_path,file_name), point_cloud_image)

    return point_cloud_image


def point_pixel_joint_matrix(scene_cloud_points):
    """

    :param scene_cloud_points:
    :return:
    """
    # Camera intrinsics  parameters
    width, height, fx, fy, cx, cy = camera_intrinsic_parameter()

    # Create an numpy array with the sane height and width of the scene image with initial value of 255
    pt_cloud_indices = - np.ones((height, width,1) , dtype=float)

    # For each point, calculate the pixel location and set the RGB values
    for idx, pt in enumerate(scene_cloud_points):
        u = int(((pt[0] * fx) / pt[2]) + cx)
        v = int(((pt[1] * fy) / pt[2]) + cy)

        pt_cloud_indices[v][u] = idx

    return pt_cloud_indices


def object_point_cloud(mask_img_file_path, scene_point_cloud_file_path):
    """
    Find the object point cloud from the joint of its mask image and scene point cloud file
    :param mask_img_file_path: Path to the image mask file
    :param scene_point_cloud_file_path: Path to the scene point cloud file
    :return: An array contain of object xyz
    """

    # Read the scene point cloud file
    scene_cloud_object = o3d.io.read_point_cloud(scene_point_cloud_file_path)
    # Scene point cloud array
    scene_point = (np.asarray(scene_cloud_object.points))
    scene_color = (np.asarray(scene_cloud_object.colors))

    # Reading the mask image
    mask_img = cv2.imread(mask_img_file_path)
    # Coordinates of the object
    object_coordinates = np.where(mask_img == 255)

    # Get point cloud indices
    pt_cloud_indices = point_pixel_joint_matrix(scene_point)

    object_pt_cloud = []
    object_pt_color = []
    for i, j in zip(object_coordinates[0], object_coordinates[1]):
        pt_index = int(pt_cloud_indices[i][j])
        if pt_index != -1:
            object_pt_cloud.append(scene_point[pt_index])
            object_pt_color.append(scene_color[pt_index])

    return np.asarray(object_pt_cloud), np.asarray(object_pt_color)


if __name__ == '__main__':
    c_i_p = camera_intrinsic_parameter()
    scene_path = "/home/user/Documents/Negar/TEASER Project/teaser_scene/scene1/"
    point_cloud_path = scene_path + "1592230042.734127275.pcd"
    mask_image_path = scene_path + "bleach_cleanser_mask.png"
    object_point_cloud(mask_image_path, point_cloud_path)
    print("The end")
