import time

import numpy as np
import sys
import open3d as o3d
# Import TEASER++
from main import convert_quaternion_to_rotation_matrix
from read_file import read_model_object_points, read_pose_file
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sklearn.neighbors as kn
import plotly.graph_objects as go
from plotly.offline import iplot

sys.path.append('/home/user/TEASER/TEASER-plusplus/build/python/teaserpp_python');
import teaserpp_python


NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10


models_folder_path = "/home/user/Documents/Negar/TEASER Project/models"
scene_folder_path = "/home/user/Documents/Negar/TEASER Project/teaser_scene/scene3"
object_name = "mustard_bottle"

translation_dic, rotation_q_dic = read_pose_file(scene_folder_path)
object_translation = translation_dic[object_name]
object_q_rotation = rotation_q_dic[object_name]

src = read_model_object_points(object_name, models_folder_path)
src = src[:3000, :3]


noisy_rotation_q = np.add(object_q_rotation, [0.132, -0.432, +0.234, -0.123])
noisy_translation = np.add(object_translation, [0.198, -0.342, -0.432])


estimated_rotation = convert_quaternion_to_rotation_matrix(noisy_rotation_q)
after_rotate = src.dot(estimated_rotation)
dst_array = after_rotate + noisy_translation
dst_array = np.asarray(dst_array)

xd = dst_array[:, 0]
yd = dst_array[:, 1]
zd = dst_array[:, 2]

xr = src[:, 0]
yr = src[:, 1]
zr = src[:, 2]

new_src = []
new_dst = []

pcd_tree = kn.KDTree(dst_array)

nn_num = 5

print(src.shape, " f", dst_array.shape)

dist, ind = pcd_tree.query(src, k=nn_num)

for i in range(len(ind)):
    for j in ind[i]:
        c = src[i]
        new_src.append(src[i])
        new_dst.append(dst_array[j])


new_src = np.asarray(new_src)
new_dst = np.asarray(new_dst)


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

new_dest = np.moveaxis(new_dst, -1, 0)
new_src = np.moveaxis(new_src, -1, 0)
print("dim: ", new_src.shape, new_dest.shape)

start = time.time()

solver.solve(new_src, new_dest)

solution = solver.getSolution()

end = time.time()

print("=====================================")
print("          TEASER++ Results           ")
print("=====================================")

print("Object name: ", object_name)
print("Estimated rotation: ")
estimated_rotation = solution.rotation
print(estimated_rotation)

print("Estimated translation: ")
estimated_translation = solution.translation
print(estimated_translation)

print("Estimated time (s): ")
print(str(end - start))

print("sldbh ")