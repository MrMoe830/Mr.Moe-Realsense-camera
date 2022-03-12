"""简化版点云测试
    参考官方教程
"""
import pyrealsense2 as rs
import numpy as np
from cv2 import cv2


def project(vert):
    """
        3D矢量投影到2D，
        我参考的白皮书投影方法:
            https://dev.intelrealsense.com/docs/projection-texture-mapping-and-occlusion-with-
            intel-realsense-depth-cameras#section-6-appendix-1-distortion-models
    """
    h, w = output.shape[:2]  # 默认宽高 848*480
    proj = vert[:, :-1] / (vert[:, -1, np.newaxis] + 0.5) * (float(h), float(h)) + (w / 2, h / 2)
    return proj


def pointcloud(out, proj, texcoords, color_img):
    """点云可视化"""
    h, w = out.shape[:2]
    j, i = proj.astype(np.uint32).T

    # 掩码，滤掉范围之外的数据
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm
    color_h, color_w = color_img.shape[:2]
    v, u = (texcoords * (color_w, color_h)).astype(np.uint64).T
    # (0,0)左上像素中心坐标，(ch-1,cw-1)右下像素中心坐标
    np.clip(u, 0, color_h - 1, out=u)   # 修改不合法值，这一步有失真
    np.clip(v, 0, color_w - 1, out=v)
    out[i[m], j[m]] = color_img[u[m], v[m]]     # 根据掩码给out图赋值符合条件的值


ctx = rs.context()
if __name__ == '__main__':

    # points = rs.points()  # 保留最后的点云数据，使其不至于随着最后一帧被释放而无数据显示

    pipe = rs.pipeline(ctx)     # 建立流传输通道
    config = rs.config()    
    # 启用流并配置需要的流属性    
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    # 开始传输    
    profile = pipe.start(config)
    # 获取指定流的配置文件，我们可以通过该配置文件访问流的相关属性
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    width_color, height_color = color_profile.width(), color_profile.height()
    width_depth, height_depth = depth_profile.width(), depth_profile.height()

    # 查看深度单位(0.001m,即默认标定一毫米)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # 为防止报错，先确定传感器是不是允许对指定的选项进行操作    
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)  # enable ir emitter
    # print('深度传感器scale为:\t{:.3f}m'.format(depth_scale))

    # # 与rgb图对齐
    algin_to = rs.align(rs.stream.color)
    pc = rs.pointcloud()    # 调用点云
    
    decimate = rs.decimation_filter(1)  # 抽取滤波器，图像降采样
    
    # 空间滤波器调参
    spatial_filter = rs.spatial_filter()    
    spatial_filter.set_option(rs.option.filter_magnitude, 2)  
    spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.25)  #
    spatial_filter.set_option(rs.option.filter_smooth_delta, 50)
    spatial_filter.set_option(rs.option.holes_fill, 3)
    
    temporal_filter = rs.temporal_filter()
    hole_fill = rs.hole_filling_filter(0)

    output = np.zeros([height_depth, width_depth, 3], dtype=np.uint8)  # 预定义点云数据可视化框
    colorizer = rs.colorizer()  # 着色
    while True:
        frames = pipe.wait_for_frames()     # 等待一组可用帧到达
        algin_frames = algin_to.process(frames) # 帧对齐

        depth_frame = frames.get_depth_frame()  # 从frameset中获取深度帧
        color_frame = frames.get_color_frame()
        # depth_frame = algin_frames.get_depth_frame()
        # color_frame = algin_frames.get_color_frame()
        
        # 图像后处理        
        depth_frame = decimate.process(depth_frame)
        depth_frame = spatial_filter.process(depth_frame)
        # depth_frame = hole_fill.process(depth_frame)
        # depth_frame = temporal_filter.process(depth_frame)
        
        # 深度帧着色，并转换为可视化图像        
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()    # 深度框架内参
        w, h = depth_intrinsics.width, depth_intrinsics.height
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)  # 告诉点云，映射是根据彩色帧元进行的

        v = points.get_vertices()   
        t = points.get_texture_coordinates()  # uv-map
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # 点云数据
        textcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  

        # 维度变换
        verts = project(verts)

        # 点云可视化
        tmp = np.zeros([h, w, 3], dtype=np.uint8)
        pointcloud(tmp, verts, textcoords, color_image)
        tmp = cv2.resize(tmp, output.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)   # 我的降采样步幅是1，所以实际没起作用，视情况调参
        # print(tmp.shape)
       
        # 可视化        
        cv2.namedWindow('Pointcloud_Test', flags=cv2.WINDOW_NORMAL)    
        cv2.imshow('Pointcloud_Test', tmp)
        # cv2.imshow('color',color_image)
        # cv2.imshow('depth image', depth_image)
        # cv2.imshow('textcoords', textcoords)
        key = cv2.waitKey(1)
        # 27->ESC键，cv2.getWindowProperty('Pointcloud_Test', cv2.WND_PROP_AUTOSIZE) < 0 作用是显示框右上角的叉，当点击×时返回负数从而终止循环       
        if key in (27, ord('q')) or cv2.getWindowProperty('Pointcloud_Test', cv2.WND_PROP_AUTOSIZE) < 0:
            print('终止播放！')
            break
    config.disable_all_streams()
    pipe.stop()
    cv2.destroyAllWindows()
