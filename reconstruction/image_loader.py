import os
from PIL import Image
import cv2
import calib_loader as cal
import open3d_viewer as ov
import numpy as np

#method = "Stanf"
method = "Kule"

if method == "Stanf":
    depth_image_dir = "../data/data_depth_completion/val_Stanf"
elif method == "Kule":
    depth_image_dir = "../data/data_depth_completion/val_kule"
raw_image_dir = "../data/data_depth_velodyne/val"
rgb_image_dir = "../data/data_rgb/val"

# foldername = ["2011_09_26_drive_0002","2011_09_26_drive_0005","2011_09_26_drive_0013",
#     "2011_09_26_drive_0020","2011_09_26_drive_0023","2011_09_26_drive_0036","2011_09_26_drive_0079",
#     "2011_09_26_drive_0095","2011_09_26_drive_0113","2011_09_26_drive_0119","2011_09_28_drive_0037",
#     "2011_09_28_drive_0225","2011_09_29_drive_0026","2011_09_29_drive_0108","2011_09_30_drive_0016",
#     "2011_09_30_drive_0072","2011_10_03_drive_0047","2011_10_03_drive_0058"]

foldername = ["2011_09_26_drive_0002","2011_09_26_drive_0005","2011_09_26_drive_0013",
    "2011_09_26_drive_0020","2011_09_26_drive_0023","2011_09_26_drive_0036","2011_09_26_drive_0079",
    "2011_09_26_drive_0095","2011_09_26_drive_0113","2011_09_28_drive_0037",
    "2011_09_29_drive_0026","2011_09_30_drive_0016","2011_10_03_drive_0047"]

file_num = [67,144,134,76,464,793,90,258,77,79,148,269,827]

def get_depth_image_path(folder_num, frame_num, cam_num = 2):
    frame_name = str(frame_num).rjust(10,'0') + ".png"
    depth_image_path = depth_image_dir + "/" + foldername[folder_num] + "/image0" + str(cam_num) + "/" + frame_name
    return depth_image_path

def get_raw_image_path(folder_num, frame_num, cam_num = 2):
    frame_name = str(frame_num).rjust(10,'0') + ".png"
    raw_image_path = raw_image_dir + "/" + foldername[folder_num] + "_sync/proj_depth/velodyne_raw/image_0" + str(cam_num) + "/" + frame_name
    return raw_image_path

def get_rgb_image_path(folder_num, frame_num, cam_num = 2):
    frame_name = str(frame_num).rjust(10,'0') + ".png"
    rgb_image_path = rgb_image_dir + "/" + foldername[folder_num] + "_sync/image_0" + str(cam_num) + "/data/" + frame_name
    return rgb_image_path

def open_depth_image(folder_num, frame_start, frame_count, cam_num = 2):
    depth_image_out = []
    if frame_start < 5 or (frame_count + frame_start) > (file_num[folder_num] + 5):
        print("There is only {} images in the folder {} !\n".format(file_num[folder_num], foldername[folder_num]))
        return None
    for i in range(frame_start, frame_start + frame_count):
        depth_image_path = get_depth_image_path(folder_num, i, cam_num)
        image_i = Image.open(depth_image_path)
        depth_image_out.append(image_i)
    return depth_image_out

def open_raw_image(folder_num, frame_start, frame_count, cam_num = 2):
    raw_image_out = []
    if frame_start < 5 or (frame_count + frame_start) > (file_num[folder_num] + 5):
        print("There is only {} images in the folder {} !\n".format(file_num[folder_num], foldername[folder_num]))
        return None
    for i in range(frame_start, frame_start + frame_count):
        raw_image_path = get_raw_image_path(folder_num, i, cam_num)
        image_i = Image.open(raw_image_path)
        raw_image_out.append(image_i)
    return raw_image_out

def open_rgb_image(folder_num, frame_start, frame_count, cam_num = 2):
    rgb_image_out = []
    if frame_start < 5 or (frame_count + frame_start) > (file_num[folder_num] + 5):
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

def Get_video_name(folder_num, frame_start, frame_count, cam_num = 2):
    filename = "folder_" + str(folder_num) + "_start_" + str(frame_start) + "_num_" + str(frame_count) + ".mp4"
    return filename

def Make_Image_Folder(folder_num, frame_start, frame_count, cam_num = 2):
    folder_name = "folder_" + str(folder_num) + "_start_" + str(frame_start) + "_num_" + str(frame_count)
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
    pcv.LoadViewFile("MainViewPoint.json")
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

DepthRgbFolder = "../data/data_depth_rgb"

def GetDepthRgbImagePath(folder_num, frame_num, cam_num = 2):
    image_folder = DepthRgbFolder + "/" + foldername[folder_num] + "_sync/image_0" + str(cam_num)
    image_name = str(frame_num).rjust(10,'0') + ".png"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image_dir = image_folder + "/" + image_name
    return image_dir

def ConvertDepthImages(folder_num, frame_start, frame_count = 0, cam_num = 2):
    if frame_count == 0:
        frame_count = file_num[folder_num]
    for frame_num in range(frame_start, frame_start + frame_count):
        print("Converting {}/{}...".format(frame_num - frame_start + 1, frame_count))
        depth_image_dir = get_depth_image_path(folder_num, frame_num, cam_num)
        dest_dir = GetDepthRgbImagePath(folder_num, frame_num, cam_num)
        ConvertDepthToRgb(depth_image_dir, dest_dir)



def ConvertDepthToRgb(image_path, dest_path):
    depth_img = cv2.imread(image_path,-1)
    #depth_img = Image.open(image_path)
    #rgb_img = np.zeros(shape=[depth_img.height, depth_img.width,3])
    #rgb_img = np.zeros(shape=[depth_img.height, depth_img.width,3])
    depth_img = np.array(depth_img, dtype=int)
    depth_img = (depth_img/256).astype(np.uint8)
    depth_img = (depth_img*256/(depth_img.max()-depth_img.min())).astype(np.uint8)
    # depth_img_f = 16777216*depth_img.astype(np.float)/65536
    # rgb_img[:,:,0] = (depth_img_f.astype(np.float)/65536).astype(int)
    # rgb_img[:,:,1] = (depth_img_f.astype(np.float)/256 - rgb_img[:,:,0]*256).astype(int)
    # rgb_img[:,:,2] = (depth_img_f.astype(np.float) - rgb_img[:,:,1]*256 - rgb_img[:,:,0]*65536).astype(int)
    rgb_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    cv2.imwrite(dest_path, rgb_img)
    # cv2.imshow("test", rgb_img)
    # cv2.waitKey(0)


#test code
# open_depth_image(0,5,20)
# open_raw_image(0,5,20)
# open_rgb_image(0,5,20)

#EncodeImagefromView(0,5,20)