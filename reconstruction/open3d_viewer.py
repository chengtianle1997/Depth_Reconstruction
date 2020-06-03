import os
import numpy as np
import cv2
import open3d as o3d

class PointCloudViewer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window('Open3D_Visualizer', 3840, 2160)
        self.opt = self.vis.get_render_option()
        #set background : black
        self.opt.background_color = np.asarray([0,0,0])
        #Basic settings
        self.scalerate = 1
        self.viewctr = self.vis.get_view_control()

    # Note: xyz is in shape (n,3), rgb is in shape (n,3)
    def ShowPointCloud(self, xyz, rgb):
        pcd_shown = o3d.geometry.PointCloud()
        pcd_shown.points = o3d.utility.Vector3dVector(xyz*self.scalerate)
        pcd_shown.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.add_geometry(pcd_shown)
        self.vis.run()
        #cv2.waitKey(0)
    
    #Get and save view point
    def SaveViewPoint(self, xyz, rgb):
        pcd_shown = o3d.geometry.PointCloud()
        pcd_shown.points = o3d.utility.Vector3dVector(xyz*self.scalerate)
        pcd_shown.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.add_geometry(pcd_shown)
        self.vis.run()

    def LoadViewFile(self, filename):
        self.viewparam = o3d.io.read_pinhole_camera_parameters(filename)

    def ShowPointCloudView(self, xyz, rgb):
        pcd_shown = o3d.geometry.PointCloud()
        pcd_shown.points = o3d.utility.Vector3dVector(xyz*self.scalerate)
        pcd_shown.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.add_geometry(pcd_shown)
        self.viewctr.convert_from_pinhole_camera_parameters(self.viewparam)
        self.vis.run()

    def NoBlockInit(self):
        self.pcd_shown = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_shown)

    def NoBlockPointCloud(self, xyz, rgb):
        self.pcd_shown.points = o3d.utility.Vector3dVector(xyz*self.scalerate)
        self.pcd_shown.colors = o3d.utility.Vector3dVector(rgb)
        self.vis.add_geometry(self.pcd_shown)
        self.viewctr.convert_from_pinhole_camera_parameters(self.viewparam)
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def CaptureImage(self, path, i):
        #image = self.vis.capture_screen_float_buffer()
        filedir = path + "/image{}.jpg".format(i)
        self.vis.capture_screen_image(filedir)
        #return np.asarray(image).astype(np.uint16)
        #return np.asarray(image)
        return filedir

    #Note: pcd is a list of o3d.geometry.PointCloud()
    def ShowMultPointCloud(self, pcd):
        apcd = o3d.geometry.PointCloud()
        for i in range(0, len(pcd)):
            apcd += pcd[i]
        self.vis.add_geometry(apcd)
        self.vis.run()

    #Filename Generator
    def FileNameGenerator(self, folder_num, frame_start, frame_count, cam_num = 2):
        filename = "folder_" + str(folder_num) + "_start_" + str(frame_start) + "_num_" + str(frame_count) + ".pcd"
        return filename

    #Save Point Cloud   
    #Note: pcd is a list of o3d.geometry.PointCloud()
    def SavePointCloud(self, pcd, filename):
        print("Saving Point Cloud......\n")
        apcd = o3d.geometry.PointCloud()
        for i in range(0, len(pcd)):
            apcd += pcd[i]
        o3d.io.write_point_cloud(filename, apcd)
        print("Point Cloud {} saved.\n".format(filename))
    
    #Read Point Cloud
    def ReadPointCloud(self, filename):
        apcd = o3d.io.read_point_cloud(filename)
        return apcd
    #Read and Show Point Cloud
    def ReadAndShowPointCloud(self, filename):
        apcd = o3d.io.read_point_cloud(filename)
        self.vis.add_geometry(apcd)
        self.vis.run()


    