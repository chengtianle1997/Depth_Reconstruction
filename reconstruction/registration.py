import open3d as o3d
import numpy as np
import copy
import image_loader as iml
import calib_loader as cal
import colored_registration as creg

calib = cal.CalibLoader()

def draw_transform_result(source, target, transform):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transform)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def Registration_piont2point(source, target):
    threshold = 1
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, 1]])
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    #draw_transform_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation

def Registration_piont2plane(source, target):
    threshold = 1
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0], [0, 0, 0, 1]])
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, trans_init)
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPlane())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    #draw_transform_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation

#Generate Point Cloud
def GenPointCloud(rgb_image, depth_image):
    xyz, rgb = calib.trans_2D_to_3D_rgbd(rgb_image, depth_image)
    res_pc = o3d.geometry.PointCloud()
    res_pc.points = o3d.utility.Vector3dVector(xyz)
    res_pc.colors = o3d.utility.Vector3dVector(rgb)
    return res_pc

def GenPointCloud_Raw(raw_image):
    xyz, color = calib.trans_2D_to_3D_depth(raw_image)
    res_pc = o3d.geometry.PointCloud()
    res_pc.points = o3d.utility.Vector3dVector(xyz)
    return res_pc

#Multiple Point Cloud Registration  ICP based on LIDAR point cloud
def MultPointCloudReg(folder_num, frame_start, frame_count, cam_num = 2):
    #Get image list
    raw_image_list, rgb_image_list, depth_image_list = iml.Get_image_list(folder_num, frame_start, frame_count, cam_num)
    image_num = len(raw_image_list)
    #Result raw point cloud list
    ResRawList = []
    #Result RGB point cloud list
    ResRGBList = []
    #Init list
    first_raw = GenPointCloud_Raw(raw_image_list[0])
    first_rgb = GenPointCloud(rgb_image_list[0], depth_image_list[0])
    ResRawList.append(first_raw)
    ResRGBList.append(first_rgb)
    skip = 2
    counter = 0
    #Regist all point clouds to image0
    for i in range(1, image_num):
        if i % skip == 0:
            #Get Point Cloud from image
            raw_pc = GenPointCloud_Raw(raw_image_list[i])
            rgb_pc = GenPointCloud(rgb_image_list[i], depth_image_list[i])
            #Find the transform to the last point cloud
            transform = Registration_piont2point(raw_pc, ResRawList[counter-1])
            counter = counter + 1
            #Transform this raw and rgb point cloud
            raw_pc.transform(transform)
            rgb_pc.transform(transform)
            #Save the transformed raw and rgb point cloud
            ResRawList.append(raw_pc)
            ResRGBList.append(rgb_pc)
            print("Registration Processing: {}/{}\n".format(i+1,image_num))
    return ResRGBList, ResRawList

#Multiple Point Cloud Registration  Colored matching
def MultPointCloudReg_cm(folder_num, frame_start, frame_count, cam_num = 2):
    #Get image list
    raw_image_list, rgb_image_list, depth_image_list = iml.Get_image_list(folder_num, frame_start, frame_count, cam_num)
    image_num = len(raw_image_list)
    #Result raw point cloud list
    #ResRawList = []
    #Result RGB point cloud list
    ResRGBList = []
    #Init list
    #first_raw = GenPointCloud_Raw(raw_image_list[0])
    first_rgb = GenPointCloud(rgb_image_list[0], depth_image_list[0])
    #ResRawList.append(first_raw)
    ResRGBList.append(first_rgb)
    counter = 1
    skip = 1
    #Regist all point clouds to image0
    for i in range(1, image_num):
        if i % skip == 0:
            print("\nRegistration Processing: {}/{}".format(i+1,image_num))
            #Get Point Cloud from image
            rgb_pc = GenPointCloud(rgb_image_list[i], depth_image_list[i])
            #Find the transform to the last point cloud
            transform = creg.colored_pointcloud_reg(rgb_pc, ResRGBList[counter-1])
            counter = counter + 1
            #Transform this rgb point cloud
            rgb_pc.transform(transform)
            #Save the transformed rgb point cloud
            ResRGBList.append(rgb_pc)
    return ResRGBList


