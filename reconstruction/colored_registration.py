import numpy as np
import open3d as o3d
import copy
from open3d.open3d.geometry import voxel_down_sample,estimate_normals

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

def colored_pointcloud_reg(source, target):
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("1/3. Downsample with a voxel size %.2f" % radius)
        #source_down = source.voxel_down_sample(radius)
        #target_down = target.voxel_down_sample(radius)
        source_down = voxel_down_sample(source, radius)
        target_down = voxel_down_sample(target, radius)

        print("2/3. Estimate normal.")
        # source_down.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        # target_down.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        estimate_normals(source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        estimate_normals(target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3/3. Applying colored point cloud registration")
        result_icp = o3d.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                    relative_rmse=1e-6,
                                                    max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    # draw_registration_result_original_color(source, target,
    #                                         result_icp.transformation)
    return result_icp.transformation

