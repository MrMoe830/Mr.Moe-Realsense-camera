import time
import pyrealsense2 as rs
import numpy as np
from cv2 import cv2


class angle_pose:
    """
    Calculate the spatial pose of the QR code

    :return: Angle with the coordinate axis or vertices of QR.
    :rtype: numpy.ndarrpy([1x3],float) or numpy,ndarray([4x2])

    - ``angle_pose(color_img,weighted_img,depth_intrin)`` is the best calling
    form to solve the pose problem in three-dimensional space.

    - ``angle_pose(color_img,weighted_img)`` can be used to calculate
    the vertex coordinates of the QR code.
    """

    def __init__(self, *args):
        self.color_img = args[0]
        self.weighted_img = args[1]
        self.depth_intrinsics = args[2] if len(args) >= 3 else None

    def QR_test(self, color_img=None, weighted_img=None, tpoint=None):
        """
        :param color_img: The color image obtained by the RGB camera
        :type color_img: ndarray(w,h,n)
        :param weighted_img: Image after stacking RGB color image and depth image
        :type weighted_img: ndarray(w,h,n)
        :param tpoint: For testing
        :return: QR_vertices
        :rtype: QR_vertices: ndarray([4x2],np.uint64)
        .. note::
            - The data is changed from float type to uint64 type, so there will be errors.
            Here, rounding is used to reduce the error
        """
        self.color_img = color_img if color_img is not None else self.color_img
        self.weighted_img = weighted_img if weighted_img is not None else self.weighted_img
        gray_image = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)
        QRdetector = cv2.QRCodeDetector()
        # cv2.circle(weighted_img, tpoint, 3, [0, 255, 0], -1, 1)
        success, QRpoints = QRdetector.detect(gray_image)
        if success:
            print(QRpoints)
            [self._qrpoints] = np.asanyarray(QRpoints + 0.5).astype(np.uint64)  # qrpoints[x->列，y->行]
            for point in self._qrpoints:
                cv2.circle(self.weighted_img, point, 3, [0, 255, 0], -1, 1)
                cv2.circle(self.color_img, point, 3, [0, 255, 0], -1, 1)  # 第200列，400行[200,400]
            return self._qrpoints
        else:
            raise AttributeError("图片属性不存在,数据缺失！")

    # noinspection PyShadowingNames
    @property
    def coordinate_3D(self):
        """3D坐标"""
        pixels = self._qrpoints
        pointcloud.map_to(color_frame)
        points = pointcloud.calculate(depth_frame)
        vertices = points.get_vertices()
        vertices = np.asanyarray(vertices).view(np.float32).reshape(-1, 3)
        xyz_point = []
        for pixel in pixels:
            i = int(self.depth_intrinsics.width * pixel[1] + pixel[0])
            xyz_point.append([vertices[i][0], vertices[i][1], vertices[i][2]])
        xyz_point = np.array(xyz_point)
        if xyz_point.any():
            raise AttributeError(f'xyz_point中含无效数据！\n\t{xyz_point}')
        else:
            return xyz_point

    @property
    def normal_vector(self):
        """二维码平面法向量"""
        vector_1 = self.coordinate_3D[2] - self.coordinate_3D[0]
        vector_2 = self.coordinate_3D[3] - self.coordinate_3D[1]
        self.__nor_vector = np.cross(vector_1, vector_2)
        return self.__nor_vector

    def angle_pose(self, norvector):
        """余弦定理求xyz欧拉角,有效角度 ±90°"""
        deg = 180 / np.pi
        # rotation = np.eye(3)
        self.__x_vector = np.array([1, 0, 0])   # x轴单位向量
        self.__y_vector = np.array([0, 1, 0])
        self.__z_vector = np.array([0, 0, 1])
        Xdot_product, Ydot_product = norvector.dot(self.__x_vector), norvector.dot(self.__y_vector)   # 点积
        Zdot_product = norvector.dot(self.__z_vector)
        angle_radx = np.arccos(
            Xdot_product / (np.sqrt(norvector.dot(norvector)) * np.sqrt(self.__x_vector.dot(self.__x_vector))))
        angle_rady = np.arccos(
            Ydot_product / (np.sqrt(norvector.dot(norvector)) * np.sqrt(self.__y_vector.dot(self.__y_vector))))
        angle_radz = np.arccos(
            Zdot_product / (np.sqrt(norvector.dot(norvector)) * np.sqrt(self.__z_vector.dot(self.__z_vector))))
        angle_deg = [90. - angle_radx * deg, 90. - angle_rady * deg, 90. - angle_radz * deg]
        return angle_deg


if __name__ == '__main__':
    time.sleep(3)
    ctx = rs.context()
    [device] = ctx.query_devices()
    depth_Sensor = device.first_depth_sensor()
    depth_Sensor.set_option(rs.option.laser_power, 360)   # 提高红外发射器功率，提高深度图像填充率和可信度
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)  # 启用depth、color流
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)

    pointcloud = rs.pointcloud()    # 点云
    points = rs.points()    # 释放帧后，保留最后的点云数据

    pipeline = rs.pipeline(ctx)
    colorizer = rs.colorizer()
    aligned = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    color_Sensor = profile.get_device().query_sensors()[1]
    # max_exp = color_Sensor.get_option_range(rs.option.exposure).max
    color_Sensor.set_option(rs.option.exposure, 70)   # 曝光时长，根据环境及需求挑即可

    # 20帧缓冲，防止传感器捕捉帧集受曝光调制时的影响
    for i in range(20):
        pipeline.wait_for_frames()

    frameset = pipeline.wait_for_frames()
    align_frameset = aligned.process(frameset)  # 帧对齐
    if frameset.size() != 2:
        raise IndexError('Missing frame data index!')
    pipeline.stop()

    # color_frame = frameset.get_color_frame()
    # depth_frame = frameset.get_depth_frame()
    color_frame = align_frameset.get_color_frame()
    depth_frame = align_frameset.get_depth_frame()

    # 获取内参
    depth_scale = depth_frame.get_units()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    # [x, y] = depth_intrin.width // 2, depth_intrin.height // 2  # for test

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    weighted = 0.5  # rgb图权重
    image = (color_image * weighted + depth_image * (1 - weighted)).astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())   # 归一化

    anglepose = angle_pose(color_image, image, depth_intrin)
    # qr_points = anglepose.QR_test()
    xyz_point = anglepose.coordinate_3D  # 二维码顶点的三维坐标
    angle = anglepose.angle_pose(anglepose.normal_vector)
    print(f'二维码3D坐标为 :\n\t{xyz_point}\n空间姿态-angle:\n\t{angle} deg')
    cv2.namedWindow("weighted_image", flags=cv2.WINDOW_NORMAL)
    # cv2.namedWindow("color_weighted_images", flags=cv2.WINDOW_NORMAL)
    # color_weighted_images = np.hstack((color_image, image))
    # cv2.imshow('color_weighted_images', color_weighted_images)
    cv2.imshow('depth', depth_image)
    cv2.imshow('weighted_image', image)
    cv2.imshow('color_image', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
