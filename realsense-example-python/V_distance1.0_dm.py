"""采用双通道,若单通道同时启用三种或更多数据流，可能会导致程序崩溃或接收不到帧数据"""
import pyrealsense2 as rs
import numpy as np
from cv2 import cv2

global imu_pipe
ctx = rs.context()

# # ---------------Parameter setting--------------- # #
DEVICE_ID = None  # None or '142422250064'
ENABLE_IMU = True  # bool

if __name__ == '__main__':
    # rgb和depth通道
    pipe = rs.pipeline(ctx)
    config = rs.config()

    if DEVICE_ID:
        config.enable_device(DEVICE_ID)
    
    # 获取设备信息以便调整分辨率设置    
    pipe_wrapper = rs.pipeline_wrapper(pipe)
    pipe_profile = config.resolve(pipe_wrapper)
    device = pipe_profile.get_device()
    device_product_line = device.get_info(rs.camera_info.product_line)
    # print(device_product_line)

    # 配置并启用指定流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(config)
    colorizer = rs.colorizer()  # colormap

    # IMU通道
    if ENABLE_IMU:
        imu_pipe = rs.pipeline(ctx)
        imu_config = rs.config()
        if DEVICE_ID:
            imu_config.enable_device(DEVICE_ID)
        imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
        imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # 陀螺仪
        imu_profile = imu_pipe.start(imu_config)
        print(imu_profile.get_stream(rs.stream.gyro, ).as_motion_stream_profile().get_motion_intrinsics().data)

    # 前十帧缓冲，主要是留给相机的曝光、适应环境的时间
    for i in range(10):
        pipe.wait_for_frames(500 if i >= 1 else 10000)

    align_to = rs.align(rs.stream.color)  # 对齐到RGB图像
    while True:
        frames = pipe.wait_for_frames()     # 等待一组可用帧到达
        align_frames = align_to.process(frames)
        depth_frames = align_frames.get_depth_frame() 
        
        # 将帧框架转换为numpy数组的可视化形式
        color_image = np.asanyarray(align_frames.get_color_frame().get_data())  
        depth_image = np.asanyarray(depth_frames.get_data())    # 未着色深度帧
        colorizer_depth = np.asanyarray(colorizer.colorize(depth_frames).get_data())    # 着色后的深度帧
     
        # 获取指定点的距离
        distance = depth_image[int(depth_image.shape[0] / 2), int(depth_image.shape[1] / 2)]
        print(distance)
        center = [int(depth_image.shape[1] / 2), int(depth_image.shape[0] / 2)]
        cv2.circle(color_image, center, 3, [0, 0, 255], -1, 2)
        cv2.putText(color_image, f'{distance}mm', [center[0] + 10, center[1] - 5], cv2.FONT_HERSHEY_COMPLEX, 1,
                    [255, 255, 0], 1, 2, False)
        cv2.imshow('RealSense D455', color_image)
        cv2.imshow('colorizer_depth',colorizer_depth)
        if ENABLE_IMU:
            imu_frames = imu_pipe.wait_for_frames()  # 等待一组可用运动帧到达
            ts = imu_frames.timestamp       # 帧元时间戳
            num_frame = imu_frames.get_frame_number()   # 帧号
            
            # 获取指定运动帧            
            acc_frame = imu_frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
            gyro_frame = imu_frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)
            print('IMU 参数:\n\tframe [{}] in [{}]:\n\taccel = {},\n\tgyro = {}'.format(
                num_frame, ts,
                acc_frame.as_motion_frame().get_motion_data(),
                gyro_frame.as_motion_frame().get_motion_data()))
        key = cv2.waitKey(50)
        if key == ord('q'):
            print('播放终止！')
            break
    pipe.stop()
    cv2.destroyAllWindows()
