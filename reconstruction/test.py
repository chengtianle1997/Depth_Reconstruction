import calib_loader as cal
import image_loader as iml
import open3d_viewer as ov
import registration as reg
import open3d as o3d
import copy
import colored_registration as creg
import matplotlib as plt
import cv2
import numpy as np
import nuscene_data_loader as nml

calib = cal.CalibLoader()
pcv = ov.PointCloudViewer()

#Test Nuscene
'''
depth_image = nml.open_depth_image(0,312,1)
rgb_image = nml.open_rgb_image(0,312,1)
xyz, color = nml.trans_2D_to_3D_rgbd(rgb_image[0], depth_image[0])
pcv.ShowPointCloud(xyz, color)
'''

#Calculate Depth Error
'''
depth_image = nml.open_depth_image(4,10,1)
raw_image = nml.open_raw_image(4,10,1)
rmse, mae, rmse_n, counter = nml.CalDepthErrForImage(raw_image[0], depth_image[0])
print("RMSE:{}, MAE:{}".format(rmse, mae))
'''

'''
rmse, mae = nml.CalDepthErrForImages(0,0,100) 
print("RMSE:{}, MAE:{}".format(rmse, mae))
'''

'''
nml.ConvertDepthImages(12,0,333)
'''

nml.ConvertDepthList("../Nuscenes Data\\kitti_8\\lidar_raw", "../Nuscenes Data\\kitti_8\\lidar_rgb")

#nml.ResizeDepthList("../Nuscenes Data\\mini\\depth_rgb", "../Nuscenes Data\\mini\\depth_rgb_rz")
#Nuscene Get and save view point
'''
depth_image = nml.open_depth_image(1,0,1)
rgb_image = nml.open_rgb_image(1,0,1)
xyz, color = nml.trans_2D_to_3D_rgbd(rgb_image[0], depth_image[0])
pcv.ShowPointCloud(xyz, color)
param = pcv.viewctr.convert_to_pinhole_camera_parameters()
print(param)
o3d.io.write_pinhole_camera_parameters("MainViewPointNuscene.json", param)
'''
#Encode Video Nuscene

#nml.EncodeImagefromView(1,0,50)


#Get and save view point
'''
depth_image = iml.open_depth_image(0,8,1)
rgb_image = iml.open_rgb_image(0,8,1)
xyz, color = calib.trans_2D_to_3D_rgbd(rgb_image[0], depth_image[0])
pcv.ShowPointCloud(xyz, color)
param = pcv.viewctr.convert_to_pinhole_camera_parameters()
print(param)
o3d.io.write_pinhole_camera_parameters("MainViewPoint.json", param)
'''
#Test ViewPoint 
'''
#Load view file
pcv.NoBlockInit()
pcv.LoadViewFile("MainViewPoint.json")
depth_image = iml.open_depth_image(0,8,2)
rgb_image = iml.open_rgb_image(0,8,2)
xyz, color = calib.trans_2D_to_3D_rgbd(rgb_image[0], depth_image[0])
#pcv.ShowPointCloudView(xyz, color)
pcv.NoBlockPointCloud(xyz, color)
image = pcv.CaptureImage("",0)
xyz, color = calib.trans_2D_to_3D_rgbd(rgb_image[1], depth_image[1])
#pcv.ShowPointCloudView(xyz, color)
pcv.NoBlockPointCloud(xyz, color)
image = pcv.CaptureImage("",1)
#cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = image.astype(np.uint8)
#cv2.imshow('image', image)
#cv2.waitKey(0)
#cv2.imwrite("image.png", image)
'''

#Encode Video

#iml.EncodeImagefromView(2,5,134)
'''
for i in range(3, len(iml.foldername)):
    print("Processing {}/{}".format(i,len(iml.foldername)))
    iml.EncodeImagefromView(i,5,iml.file_num[i])
'''

#Convert Depth image into colored image
'''
for i in range(1, len(iml.foldername)):
    print("-->Processing {}/{}......".format(i,len(iml.foldername)))
    iml.ConvertDepthImages(i,5)
'''
#iml.ConvertDepthToRgb("../data\\data_depth_completion\\val_kule\\2011_09_26_drive_0005\\image02\\0000000005.png", "../data/data_depth_rgb/test.png")

#Test one colored point-cloud
'''
depth_image = iml.open_depth_image(7,8,1)
rgb_image = iml.open_rgb_image(7,8,1)
xyz, color = calib.trans_2D_to_3D_rgbd(rgb_image[0], depth_image[0])
pcv.ShowPointCloud(xyz, color)
'''

#Test one raw lidar point-cloud
'''
raw_image = iml.open_raw_image(0,5,1)
xyz, color = calib.trans_2D_to_3D_depth(raw_image[0])
pcv.ShowPointCloud(xyz, color)
'''

#Test two lidar Point-cloud Registration
'''
raw_image = iml.open_raw_image(0,64,2)
xyz_0, color = calib.trans_2D_to_3D_depth(raw_image[0])
xyz_1, color = calib.trans_2D_to_3D_depth(raw_image[1])
p_xyz_0 = o3d.geometry.PointCloud()
p_xyz_1 = o3d.geometry.PointCloud()
p_xyz_0.points = o3d.utility.Vector3dVector(xyz_0)
p_xyz_1.points = o3d.utility.Vector3dVector(xyz_1)

transform = reg.Registration_piont2point(p_xyz_0, p_xyz_1)
#transform = reg.Registration_piont2plane(p_xyz_0, p_xyz_1)
# transform = [[ 9.99981291e-01,-6.09089093e-03,-5.65258468e-04,-9.13100151e-04],
#  [ 6.08922504e-03,9.99977245e-01,-2.90347640e-03,1.11673512e-03],
#  [ 5.82930363e-04,2.89998010e-03,9.99995625e-01,6.72217034e-04],
#  [ 0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]]
#reg.draw_transform_result(p_xyz_0, p_xyz_1,transform)

#Test two colored-pointcloud registration
depth_image = iml.open_depth_image(0,64,2)
rgb_image = iml.open_rgb_image(0,64,2)
rxyz_0, color_0 = calib.trans_2D_to_3D_rgbd(rgb_image[0], depth_image[0])
rxyz_1, color_1 = calib.trans_2D_to_3D_rgbd(rgb_image[1], depth_image[1])
pr_xyz_0 = o3d.geometry.PointCloud()
pr_xyz_1 = o3d.geometry.PointCloud()
pr_xyz_0.points = o3d.utility.Vector3dVector(rxyz_0)
pr_xyz_1.points = o3d.utility.Vector3dVector(rxyz_1)
pr_xyz_0.transform(transform)
pr_xyz_0.colors = o3d.utility.Vector3dVector(color_0)
pr_xyz_1.colors = o3d.utility.Vector3dVector(color_1)
# source_temp = copy.deepcopy(pr_xyz_0)
# target_temp = copy.deepcopy(pr_xyz_1)
# source_temp.transform(transform)
# source_temp.colors = o3d.utility.Vector3dVector(color_0)
# target_temp.colors = o3d.utility.Vector3dVector(color_1)
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pr_xyz_0)
# vis.run()
#pcv.ShowPointCloud(rxyz_0, color_0)
# pcv.vis.add_geometry(pr_xyz_0)
# pcv.vis.run()

pcv.vis.add_geometry(pr_xyz_0 + pr_xyz_1)
#pcv.vis.add_geometry(pr_xyz_1)
pcv.vis.run()
'''

#Test Mult Point Cloud Registration
'''
ResRGBList, ResRawList = reg.MultPointCloudReg(0, 5, 12)
#pcv.ShowMultPointCloud(ResRGBList)
filename = pcv.FileNameGenerator(0, 5, 12)
pcv.SavePointCloud(ResRGBList, filename)
pcv.ShowMultPointCloud(ResRGBList)
#pcv.ShowMultPointCloud(ResRawList)
'''
'''
filename = pcv.FileNameGenerator(7, 5, 5)
pcv.ReadAndShowPointCloud(filename)
'''

#Test Colored Point Cloud Registration
'''
raw_image_list, rgb_image_list, depth_image_list = iml.Get_image_list(7,15,2)
pc1 = reg.GenPointCloud(rgb_image_list[0],depth_image_list[0])
pc2 = reg.GenPointCloud(rgb_image_list[1],depth_image_list[1])
creg.colored_pointcloud_reg(pc1, pc2)
'''

#Test Colored Point Cloud Registration (Multi-PointCloud)
'''
ResRGBList = reg.MultPointCloudReg_cm(7, 5, 6)
#pcv.ShowMultPointCloud(ResRGBList)
filename = pcv.FileNameGenerator(7, 5, 6)
pcv.SavePointCloud(ResRGBList, filename)
pcv.ShowMultPointCloud(ResRGBList)
'''
'''
filename = pcv.FileNameGenerator(7, 5, 5)
pcv.ReadAndShowPointCloud(filename)
'''

'''
import numpy as np
xyz = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
xyz = np.array(xyz)
print(xyz)
# xyz = np.reshape(xyz,(4,1,3))
# xyz = np.reshape(xyz,(2,2,3))
# print(xyz)
index = np.where((xyz[:,2]>7)&(xyz[:,2]<10))
xyz = xyz[index]
print(index)
print(xyz)
'''