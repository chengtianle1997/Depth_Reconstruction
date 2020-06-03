import os
from PIL import Image
import cv2
import calib_loader as cal
import open3d_viewer as ov
import numpy as np
import math

dataset_folder = "../Nuscenes Data/"
scene_folder = "mini"
rgb_branch = "/rgb_image/"
lidar_branch = "/lidar_raw/"
result_branch = "/out_result/"

folder_name = ["mini", "mini_crop", "kitti_16", "kitti_32", "kitti_64", "kitti_model", "kitti_8", "kitti_4","kitti_erfnet_depth", "kitti_erfnet_rgbd", "kitti_hourglass_depth", "kitti_erfnet_hourglass", "kitti_final"]
file_num = [404, 404, 400, 400, 400, 67, 400, 400, 333, 333, 333, 333, 333]

#Camera param
cam_uo = 816.2670197447984 
cam_vo = 491.50706579294757 
#cam_uo = 816.2670197447984 - int((1600-1216)/2)
#cam_vo = 491.50706579294757 - int((900-256)/2)
cam_fx = 1266.417203046554
cam_fy = 1266.417203046554



def get_depth_image_path(folder_num, frame_num, cam_num = 2):
    frame_name = str(frame_num).rjust(10,'0') + ".png"
    depth_image_path = dataset_folder + folder_name[folder_num] + result_branch + frame_name
    return depth_image_path

def get_raw_image_path(folder_num, frame_num, cam_num = 2):
    frame_name = str(frame_num).rjust(10,'0') + ".png"
    raw_image_path = dataset_folder + folder_name[folder_num] + lidar_branch + frame_name
    return raw_image_path

def get_rgb_image_path(folder_num, frame_num, cam_num = 2):
    frame_name = str(frame_num).rjust(10,'0') + ".png"
    rgb_image_path = dataset_folder + folder_name[folder_num] + rgb_branch + frame_name
    return rgb_image_path

def open_depth_image(folder_num, frame_start, frame_count, cam_num = 2):
    depth_image_out = []
    if (frame_count + frame_start) > (file_num[folder_num]):
        print("There is only {} images in the folder {} !\n".format(file_num[folder_num], foldername[folder_num]))
        return None
    for i in range(frame_start, frame_start + frame_count):
        depth_image_path = get_depth_image_path(folder_num, i, cam_num)
        image_i = Image.open(depth_image_path)
        depth_image_out.append(image_i)
    return depth_image_out

def open_raw_image(folder_num, frame_start, frame_count, cam_num = 2):
    raw_image_out = []
    if (frame_count + frame_start) > (file_num[folder_num]):
        print("There is only {} images in the folder {} !\n".format(file_num[folder_num], foldername[folder_num]))
        return None
    for i in range(frame_start, frame_start + frame_count):
        raw_image_path = get_raw_image_path(folder_num, i, cam_num)
        image_i = Image.open(raw_image_path)
        raw_image_out.append(image_i)
    return raw_image_out

def open_rgb_image(folder_num, frame_start, frame_count, cam_num = 2):
    rgb_image_out = []
    if (frame_count + frame_start) > (file_num[folder_num]):
        print("There is only {} images in the folder {} !\n".format(file_num[folder_num], foldername[folder_num]))
        return None
    for i in range(frame_start, frame_start + frame_count):
        rgb_image_path = get_rgb_image_path(folder_num, i, cam_num)
        image_i = Image.open(rgb_image_path)
        rgb_image_out.append(image_i)
    return rgb_image_out

def Get_image_list(folder_num, frame_start, frame_count, cam_num = 2):
    raw_image_list = open_raw_image(folder_num, frame_start, frame_count, cam_num)
    rgb_image_list = open_rgb_image(folder_num, frame_start, frame_count, cam_num)
    depth_image_list = open_depth_image(folder_num, frame_start, frame_count, cam_num)
    return raw_image_list, rgb_image_list, depth_image_list

def GetROI(xyz, rgb, xs, xd, ys, yd, zs, zd):
        width = 1216
        height = 256
        ratio = 1
        m = width * height
        '''
        wh_x = np.reshape(xyz[:,0], (height, width))
        wh_y = np.reshape(xyz[:,1], (height, width))
        wh_z = np.reshape(xyz[:,2], (height, width))
        wh_r = np.reshape(rgb[:,0], (height, width))
        wh_g = np.reshape(rgb[:,1], (height, width))
        wh_b = np.reshape(rgb[:,2], (height, width))
        res_x = wh_x[ys:yd, xs:xd]
        res_y = wh_y[ys:yd, xs:xd]
        res_z = wh_z[ys:yd, xs:xd]
        res_r = wh_r[ys:yd, xs:xd]
        res_g = wh_g[ys:yd, xs:xd]
        res_b = wh_b[ys:yd, xs:xd]
        '''
        wh_xyz = np.reshape(xyz, (m,1,3))
        wh_rgb = np.reshape(rgb, (m,1,3))
        wh_xyz = np.reshape(wh_xyz, (height,width,3))
        wh_rgb = np.reshape(wh_rgb, (height,width,3))
        res_rgb = wh_rgb[ys:yd, xs:xd]
        res_xyz = wh_xyz[ys:yd, xs:xd]
        n = (yd-ys)*(xd-xs)
        res_xyz = np.reshape(res_xyz, (n,1,3))
        res_xyz = np.reshape(res_xyz, (n,3))
        res_rgb = np.reshape(res_rgb, (n,1,3))
        res_rgb = np.reshape(res_rgb, (n,3))
        index_fit = np.where((res_xyz[:,2]<zd) & (res_xyz[:,2]>zs))
        res_rgb = res_rgb[index_fit]
        res_xyz = res_xyz[index_fit]

        return res_xyz*ratio, res_rgb

def trans_2D_to_3D_rgbd(rgb_image, depth_image):
    width = 1600
    height = 900
    dep_width = 1216
    dep_height = 256
    #Init
    m = dep_width * dep_height
    #n = width * height
    n = m
    XYZ_PointCloud = np.zeros([3,m],dtype=np.float32)
    RGB_PointCloud = np.zeros([3,m],dtype=np.float32) 
    depth = np.array(depth_image,dtype=int)
    depth = depth.astype(np.float)/256
    #Color Transform
    color_arr = rgb_image.split()
    color = np.zeros([3,n],dtype=np.float32)
    color_arr_r = np.reshape(color_arr[0],(1,n))
    color_arr_g = np.reshape(color_arr[1],(1,n))
    color_arr_b = np.reshape(color_arr[2],(1,n))
    color[0,:] = color_arr_r/256
    color[1,:] = color_arr_g/256
    color[2,:] = color_arr_b/256
    #color = color.T
    #Y = np.zeros([3,m],dtype=float)
    for u in range(0, dep_height):
        for v in range(0, dep_width):
            ref = u * dep_width + v
            #ref_c = int(u * width/dep_width)* width + int(v * height/dep_height)
            ref_c = ref
            x, y = -u*width/dep_width, v*height/dep_height
            XYZ_PointCloud[:,ref] = [x,y,depth[u,v]]
            RGB_PointCloud[:,ref] = [color[0,ref_c],color[1,ref_c],color[2,ref_c]]
    #Coordinate transform
    px = XYZ_PointCloud[0,:]
    py = XYZ_PointCloud[1,:]
    depth = XYZ_PointCloud[2,:]
    x = (px - cam_uo)/cam_fx
    y = (py - cam_vo)/cam_fy
    XYZ_PointCloud[0,:] = depth * x
    XYZ_PointCloud[1,:] = depth * y
    XYZ_PointCloud = XYZ_PointCloud.T
    RGB_PointCloud = RGB_PointCloud.T
    XYZ_PointCloud, RGB_PointCloud = GetROI(XYZ_PointCloud, RGB_PointCloud, 0, 1216, 90, 256, 0, 200)
    return XYZ_PointCloud, RGB_PointCloud

depthrgb_branch = "/depth_rgb/"

def GetDepthRgbImagePath(folder_num, frame_num, cam_num = 2):
    image_folder = dataset_folder + folder_name[folder_num] + depthrgb_branch
    image_name = str(frame_num).rjust(10,'0') + ".png"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image_dir = image_folder + image_name
    return image_dir

def ConvertDepthImages(folder_num, frame_start, frame_count = 0, cam_num = 2):
    if frame_count == 0:
        frame_count = file_num[folder_num]
    for frame_num in range(frame_start, frame_start + frame_count):
        print("Converting {}/{}...".format(frame_num - frame_start + 1, frame_count))
        depth_image_dir = get_depth_image_path(folder_num, frame_num, cam_num)
        dest_dir = GetDepthRgbImagePath(folder_num, frame_num, cam_num)
        ConvertDepthToRgb(depth_image_dir, dest_dir)

def ConvertDepthList(in_folder, out_folder):
    file_list = os.listdir(in_folder)
    for i in range(len(file_list)):
        img_dir = in_folder + "/" + file_list[i]
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        img_dst = out_folder + "/" + str(i).rjust(10,'0') + ".png"
        ConvertLiDARDepth(img_dir, img_dst)
        if i % 20 == 0:
            print("Processing {}/{}".format(i, len(file_list)))

def ConvertLiDARDepth(image_path, dest_path):
    depth_img = cv2.imread(image_path,-1)
    depth_img = np.array(depth_img, dtype=int)
    depth_img = (depth_img/256).astype(np.uint8)
    depth_img = (depth_img*256/(depth_img.max()-depth_img.min())).astype(np.uint8)
    rgb_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    #rgb_img = cv2.resize(rgb_img, (1600,900))
    #rgb_img = cv2.resize(rgb_img, (1600, 900))
    cv2.imwrite(dest_path, rgb_img)
    #cv2.imshow("test", rgb_img)
    #cv2.waitKey(0)

def ResizeDepthList(in_folder, out_folder):
    file_list = os.listdir(in_folder)
    for i in range(len(file_list)):
        img_dir = in_folder + "/" + file_list[i]
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        img_dst = out_folder + "/" + str(i).rjust(10,'0') + ".png"
        ResizeLiDARDepth(img_dir, img_dst)
        if i % 20 == 0:
            print("Processing {}/{}".format(i, len(file_list)))

def ResizeLiDARDepth(image_path, dest_path):
    rgb_img = cv2.imread(image_path,-1)
    # depth_img = np.array(depth_img, dtype=int)
    # depth_img = (depth_img/256).astype(np.uint8)
    # depth_img = (depth_img*256/(depth_img.max()-depth_img.min())).astype(np.uint8)
    # rgb_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    #rgb_img = cv2.resize(rgb_img, (1600,900))
    rgb_img = cv2.resize(rgb_img, (1600, 900))
    cv2.imwrite(dest_path, rgb_img)
    #cv2.imshow("test", rgb_img)
    #cv2.waitKey(0)


def ConvertDepthToRgb(image_path, dest_path):
    depth_img = cv2.imread(image_path,-1)
    depth_img = np.array(depth_img, dtype=int)
    depth_img = (depth_img/256).astype(np.uint8)
    depth_img = (depth_img*256/(depth_img.max()-depth_img.min())).astype(np.uint8)
    rgb_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    #rgb_img = cv2.resize(rgb_img, (1600,900))
    cv2.imwrite(dest_path, rgb_img)
    #cv2.imshow("test", rgb_img)
    #cv2.waitKey(0)

def Get_video_name(folder_num, frame_start, frame_count, cam_num = 2):
    filename = "Nuscenefolder_" + str(folder_num) + "_start_" + str(frame_start) + "_num_" + str(frame_count) + ".mp4"
    return filename

def Make_Image_Folder(folder_num, frame_start, frame_count, cam_num = 2):
    folder_name = "Nuscenefolder_" + str(folder_num) + "_start_" + str(frame_start) + "_num_" + str(frame_count)
    if os.path.exists(folder_name):
        return folder_name
    os.makedirs(folder_name)
    return folder_name

def EncodeImagefromView(folder_num, frame_start, frame_count, cam_num = 2):
    video_name = Get_video_name(folder_num, frame_start, frame_count, cam_num)
    foldername = Make_Image_Folder(folder_num, frame_start, frame_count, cam_num)
    raw_image_list, rgb_image_list, depth_image_list = Get_image_list(folder_num, frame_start, frame_count, cam_num)
    num = len(rgb_image_list)
    #Image Height and Width Origin:1920*1080
    height = 2108
    width = 3840
    #Video Head Init
    fps = 18
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(foldername + "/" + video_name, fourcc, fps, (width, height))
    #Class Init
    calib = cal.CalibLoader()
    pcv = ov.PointCloudViewer()
    pcv.LoadViewFile("MainViewPointNuscene.json")
    pcv.NoBlockInit()
    for i in range(num):
        xyz, color = calib.trans_2D_to_3D_rgbd(rgb_image_list[i], depth_image_list[i])
        #pcv.vis.create_window()
        #pcv.ShowPointCloudView(xyz, color)
        pcv.NoBlockPointCloud(xyz, color)
        imagedir = pcv.CaptureImage(foldername, i)
        image = cv2.imread(imagedir)
        #image = cv2.resize(image,(width,height))
        video_writer.write(image)
        #pcv.vis.destroy_window()
        print("Encoding:{}/{}\n".format(i, frame_count))
    video_writer.release()
    print("Encode process finished!\n")

def CalDepthErrForImage(raw_image, depth_image):
    depth = np.array(depth_image,dtype=int)
    raw = np.array(raw_image,dtype=int)
    depth = depth.astype(np.float)/256
    raw = raw.astype(np.float)/256
    raw_i = np.argwhere(raw>0)
    raw_num = len(raw_i)
    height, width = depth.shape[0], depth.shape[1]
    counter = 0
    rmse_n = 0
    mae_n = 0
    rand_num = 1000
    rand = int(raw_num/rand_num)
    rand_count = 0
    for i in range(height):
        for j in range(width):
            if not raw[i,j] == 0: #and abs(raw[i,j]-depth[i,j])<20:
                rand_count += 1
                if rand_count % rand == 0:
                    counter = counter + 1
                    rmse_n += (raw[i,j]-depth[i,j])*(raw[i,j]-depth[i,j])
                    mae_n += abs(raw[i,j]-depth[i,j])
    rmse = math.sqrt(rmse_n/counter)
    mae = mae_n/counter
    return rmse, mae, rmse_n, counter
    
def CalDepthErrForImages(folder_num, frame_start, frame_count, cam_num = 2):
    rmse = 0
    mae = 0
    raw_image_list, rgb_image_list, depth_image_list = Get_image_list(folder_num, frame_start, frame_count, cam_num)
    n = len(raw_image_list)
    counter = 0
    for i in range(n):
        depth_image, raw_image = depth_image_list[i], raw_image_list[i]
        rmse_i, mae_i, rmse_n, num =  CalDepthErrForImage(raw_image, depth_image)
        rmse += rmse_i
        mae += mae_i
        counter += num
        if i%20 == 0:
            print("processing {}/{}\n".format(i, n))
    rmse = rmse/n
    mae = mae/n
    return rmse, mae

