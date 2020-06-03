import os
import numpy as np
from PIL import Image
import image_loader as iml

class CalibLoader:
    #cam_num = 2 or 3
    def __init__(self, cam_num = 2, date = "2011_09_26"):
        #calib file path
        self.calib_path = "../calibration/" + date + "/calib_cam_to_cam.txt"
        #calib param init
        self.Prect = np.zeros([3,4],dtype=float)
        self.uo = 0
        self.vo = 0
        self.fx = 0
        self.fy = 0

        #calib load
        calibtxt = open(self.calib_path, "r")
        lines = calibtxt.readlines()
        if cam_num == 2:
            line_num = 25
        elif cam_num == 3:
            line_num = 33
        else:
            print("We do not have this camera!\n")
            return
        P_rect_str = lines[line_num].split(":")[1].split(" ")[1:]
        self.Prect = np.reshape(np.array([float(p) for p in P_rect_str]),(3,4)).astype(np.float32)
        self.uo = self.Prect[0,2]
        self.vo = self.Prect[1,2]
        self.fx = self.Prect[0,0]
        self.fy = self.Prect[1,1]
        self.ratio = 1000
        self.isfirst = True
    
    def trans_2D_to_3D_rgbd(self, rgb_image, depth_image):
        width = depth_image.size[0]
        height = depth_image.size[1]
        #Save standard width and height
        self.width = width
        self.height = height
        if not (rgb_image.size[0] == width and rgb_image.size[1] == height):
            #print("The RGB image and Depth image are not in the same size!")
            #return None
            tw = depth_image.size[0]
            th = depth_image.size[1]
            sw = 0
            sh = 0
            if iml.method == "Kule":
                sw = 0
                sh = int(rgb_image.size[1]-depth_image.size[1])
            elif iml.method == "Scanf":
                sw = int((rgb_image.size[0]-depth_image.size[0])/2)
                sh = int((rgb_image.size[1]-depth_image.size[1])/2)
            #sw = int(self.uo-depth_image.size[0]/2)
            #sh = int(self.vo-depth_image.size[1]/2)
            rgb_image = rgb_image.crop((sw,sh,sw+tw,sh+th))
            if self.isfirst:
                self.uo = self.uo - sw
                self.vo = self.vo - sh
                self.isfirst = False
        n = width*height
        XYZRGB_PointCloud = np.zeros([4,n],dtype=np.float32)
        depth = np.array(depth_image,dtype=int)
        depth = depth.astype(np.float)/256
        #color_arr = np.array(rgb_image,dtype=np.uint32)
        #Color transform
        color_arr = rgb_image.split()
        color = np.zeros([3,n],dtype=np.float32)
        color_arr_r = np.reshape(color_arr[0],(1,n))
        color_arr_g = np.reshape(color_arr[1],(1,n))
        color_arr_b = np.reshape(color_arr[2],(1,n))
        color[0,:] = color_arr_r/256
        color[1,:] = color_arr_g/256
        color[2,:] = color_arr_b/256
        color = color.T
        Y = np.zeros([3,n],dtype=float)
        #Coordinate transform
        for u in range(0,height):
            for v in range(0,width):
                m = u*width + v
                Y[:,m] = [-u,v,depth[u][v]]
        px = Y[0,:]
        py = Y[1,:]
        depth = Y[2,:] 
        x = (px - self.uo)/self.fx
        y = (py - self.vo)/self.fy
        Y[0,:] = depth * x
        Y[1,:] = depth * y
        #output pointcloud
        # XYZRGB_PointCloud[0:3] = Y
        # XYZRGB_PointCloud[3] = color
        # XYZRGB_PointCloud = XYZRGB_PointCloud.T
        # XYZRGB_PointCloud = XYZRGB_PointCloud.astype(np.float32)
        xyz = Y.T
        #xyz, color = self.GetROI(xyz, color, int(0.1*width), int(0.9*width), int(0.05*height), int(0.95*height), 0, 50)
        return xyz, color
    
    def trans_2D_to_3D_depth(self, depth_image):
        width = depth_image.size[0]
        height = depth_image.size[1]
        n = width*height
        XYZRGB_PointCloud = np.zeros([4,n],dtype=np.float32)
        depth = np.array(depth_image,dtype=int)
        depth = depth.astype(np.float)/256
        color = np.zeros([3,n],dtype=np.float32)
        Y = np.zeros([3,n],dtype=float)
        color = color + 1
        color = color.T
        #Coordinate transform
        for u in range(0,height):
            for v in range(0,width):
                m = u*width + v
                Y[:,m] = [-u,v,depth[u][v]]
        px = Y[0,:]
        py = Y[1,:]
        depth = Y[2,:] 
        x = (px - self.uo)/self.fx
        y = (py - self.vo)/self.fy
        Y[0,:] = depth * x
        Y[1,:] = depth * y
        #output pointcloud
        xyz = Y.T
        #decrease the num of point
        de_rate = 8
        max_i = int(len(xyz)/de_rate)
        xyz = xyz[[int(de_rate*i) for i in range(0,max_i)]]
        color = color[[int(de_rate*i) for i in range(0,max_i)]]
        return xyz*self.ratio, color

    #Get the ROI of the Point Cloud  Note: x, y related to the camera plane   z is the real depth in meter
    #At the end of this function, the unit of the point cloud will be transformed from meter to millimeter
    def GetROI(self, xyz, rgb, xs, xd, ys, yd, zs, zd):
        width = self.width
        height = self.height
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

        return res_xyz*self.ratio, res_rgb


