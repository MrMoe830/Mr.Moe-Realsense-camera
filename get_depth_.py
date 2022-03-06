import time

import pyrealsense2 as rs
from cv2 import cv2
import numpy as np

if __name__ == '__main__':
    ctx = rs.context()
    [device] = ctx.devices
    ctx.devices[0].hardware_reset()
    time.sleep(5)
    sensors = ctx.sensors
    depth_sensor = sensors[0].as_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print(depth_sensor.get_stream_profiles())

    depth_sensor.set_option(rs.option.exposure, 15000)
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    depth_sensor.set_option(rs.option.laser_power, 360)
    depth_sensor.set_option(rs.option.emitter_always_on, 1)
    # depth_sensor.set_option(rs.option.laser_power, 360)
    # print(depth_sensor.get_option(rs.option.emitter_always_on))
    # print(depth_sensor.get_option_value_description(rs.option.emitter_enabled, 3))
    # print(depth_sensor.get_option_range(rs.option.laser_power))
    # print(depth_sensor.get_supported_options())

    # roi = rs.region_of_interest()
    # roi.max_x = 800
    # roi.max_y = 600
    # roi.min_x = 400
    # roi.min_y = 200
    # rs.roi_sensor(depth_sensor).set_region_of_interest(roi)
    # print(rs.roi_sensor(depth_sensor).get_region_of_interest())

    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline(ctx)
    pipeline_profile = pipeline.start(config)
    depth_pixel = [240, 320]
    while True:
        frameset = pipeline.wait_for_frames()
        # print(frameset.size())
        composite_frame = rs.composite_frame(frameset)
        depth_frame = composite_frame.first(rs.stream.depth, rs.format.z16).as_depth_frame()
        color_frame = composite_frame.first(rs.stream.color, rs.format.bgr8).as_video_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_point = rs.rs2_deproject_pixel_to_point(color_intrin, depth_pixel, depth_scale)
        depth_image = np.asanyarray(rs.colorizer().colorize(depth_frame).get_data())
        print(depth_point)
        cv2.imshow('depth', depth_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    pipeline.stop()
