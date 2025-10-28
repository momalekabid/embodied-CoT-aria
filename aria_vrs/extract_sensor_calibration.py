from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.image import InterpolationMethod
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import cv2

vrsfile = "./vrs_data/Microsoft_office_1.vrs"

provider = data_provider.create_vrs_data_provider(vrsfile)

# returns None if vrs does not have a calibration
device_calib = provider.get_device_calibration()
print(device_calib.get_device_subtype())


label = "camera-slam-right"
transform_device_sensor = device_calib.get_transform_device_sensor(label)
transform_device_cpf = device_calib.get_transform_device_cpf()
transform_cpf_sensor = device_calib.get_transform_cpf_sensor(label)

# returns None if vrs does not have a calibration
device_calib = provider.get_device_calibration()
sensor_calib = device_calib.get_sensor_calib(label)

# print(transform_cpf_sensor)
# print(transform_device_sensor)
# print(device_calib)
# print(sensor_calib)

cam_calib = device_calib.get_camera_calib("camera-slam-left")
# print(provider.get_label_from_stream_id())
calib = provider.get_device_calibration().get_camera_calib("camera-slam-left")

print(calib.get_transform_device_camera())

# print(cam_calib)
# print(type(cam_calib))
def undistort_to_linear(provider, stream_ids, raw_image, camera_label="rgb"):
    # input: retrieve image as a numpy array
    camera_name = "camera-rgb"

    sensor_name = "camera-rgb"
    sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
    image_data = provider.get_image_data_by_index(sensor_stream_id, 0)
    image_array = image_data[0].to_numpy_array()
    # input: retrieve image distortion
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)

    # create output calibration: a linear model of image size 512x512 and focal length 150
    # Invisible pixels are shown as black.
    dst_calib = calibration.get_linear_camera_calibration(512, 512, 150, camera_name)

    # distort image
    rectified_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)

    # visualize input and results
    plt.figure()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Image undistortion (focal length = {dst_calib.get_focal_lengths()})")

    axes[0].imshow(image_array, cmap="gray", vmin=0, vmax=255)
    axes[0].title.set_text(f"sensor image ({sensor_name})")
    axes[0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    axes[1].imshow(rectified_array, cmap="gray", vmin=0, vmax=255)
    axes[1].title.set_text(f"undistorted image ({sensor_name})")
    axes[1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plt.show()


stream_ids: Dict[str, StreamId] = {
        "rgb": StreamId("214-1"),
        "slam-left": StreamId("1201-1"),
        "slam-right": StreamId("1201-2"),
    }


def reproject_point(pose, provider):
    ## cam_matrix := extrinsics
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = provider.get_device_calibration()
    # point_pose_camera = cam_matrix @ pose_hom
    # print(point_pose_camera)
    calib = device_calibration.get_camera_calib(rgb_stream_label)
    T_device_sensor = device_calibration.get_transform_device_sensor(rgb_stream_label)
    point_position_camera = T_device_sensor.inverse() @ pose

    warped = calibration.get_linear_camera_calibration(
        480, 640, 50.25430222 * 2, "rgb", calib.get_transform_device_camera()
    )
    point_position_pixel = warped.project(point_position_camera)
    return point_position_pixel



raw_img = cv2.imread("./extracted/data/ms_office/214-1/214-1-00011-78.445.jpg")
img = undistort_to_linear(provider, stream_ids, raw_img) 
