######################################################################
#############   用例演示了:                             ###############
#############       从深度像素点获取对应的实际坐标值        ###############
#############       获取深度值的API调用                  ###############
######################################################################
import time
import pyrealsense2 as rs
from cv2 import cv2
import numpy as np

if __name__ == '__main__':
    ctx = rs.context()  
    [device] = ctx.devices  # 获取设备
    ctx.devices[0].hardware_reset()     # 设备重置，以防受上次调参影响
    time.sleep(5)
    sensors = ctx.sensors   # 获取传感器
    depth_sensor = sensors[0].as_depth_sensor()     # D455具有三个传感器模块，‘立体声模块’，‘RGB模块’，‘运动模块’，其中立体声模块既包含深度流又包含红外流
    depth_scale = depth_sensor.get_depth_scale()    # 获取深度单位，Intel Realsense D455默认情况下的scale为1mm
    
    # 调参    
    depth_sensor.set_option(rs.option.exposure, 15000)
    depth_sensor.set_option(rs.option.emitter_enabled, 1)   # 发射器状态，1->open 0->close
    depth_sensor.set_option(rs.option.laser_power, 360)     # 红外功率，D455最大为360
    depth_sensor.set_option(rs.option.emitter_always_on, 1) 
    # print(depth_sensor.get_option(rs.option.emitter_always_on))   # 获取参数当前值
    # print(depth_sensor.get_option_value_description(rs.option.emitter_enabled, 1))
    # print(depth_sensor.get_option_range(rs.option.laser_power))
    # print(depth_sensor.get_supported_options())

    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)      # 启用深度流
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline(ctx)     # 为设备创建流式传输管道
    pipeline_profile = pipeline.start(config)   # 配置文件
    depth_pixel = [240, 320]    # 随机像素点
    while True:
        frameset = pipeline.wait_for_frames()   # 等待一组可用帧集
        # composite_frame = rs.composite_frame(frameset)  # 用于下面检索特定流的第一帧
        # depth_frame = composite_frame.first(rs.stream.depth, rs.format.z16).as_depth_frame()
        # color_frame = composite_frame.first(rs.stream.color, rs.format.bgr8).as_video_frame()
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics   # 深度框架内参
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        
        # 注：深度图的像素数据表示的就是每个点的深度值                 
        distance = depth_frame.get_distance(depth_frame.width//2,depth_frame.height//2)      # 获取深度，与两点间距离不同,深度距离指的是三维坐标中的Z轴分量，depth_pixel
        depth_point = rs.rs2_deproject_pixel_to_point(color_intrin, depth_pixel, depth_scale)   # API解算深度图像中某个深度像素对应相机坐标系下的实际三维坐标
        depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())    # 给深度图着色并转换为array数组以显示彩色深度图像
        print(depth_point，distance, sep='\n')      # 输出对应的深度点和深度值，仔细观察可知distance就等于depth_point中的Z值
        cv2.imshow('depth', depth_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    pipeline.stop()
    cv2.destroyAllWindows()
    
