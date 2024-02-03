import sys
import cv2
import pyzed.sl as sl
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import time

import asyncio
import qasync

from perseus.detector.models import KeypointCNN
import torch
import kornia
import torchvision

from perseus.smoother.base_v2 import (
    FixedLagSmoother,
    RigidBodyTrajectory,
    SmootherConfig,
)
from perseus.smoother.utils import *
from functools import partial


class ZEDCamera:
    def __init__(self):
        # Initialize the ZED camera
        self.camera = sl.Camera()
        self.image = sl.Mat()

        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD720 video mode

        # init_params.camera_fps = 30  # Set the camera to run at 30 FPS
        init_params.set_from_serial_number(33143189)

        # Open the camera
        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

    async def get_frame_async(self):
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self.get_frame)
        return frame

    def get_frame(self):
        runtime_parameters = sl.RuntimeParameters()
        if self.camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
            frame = self.image.get_data()
            return frame
        return None

    def close(self):
        # Close the ZED camera
        self.camera.close()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, zed_camera, parent=None):
        super(MainWindow, self).__init__(parent)
        self.zed_camera = zed_camera

        # Fix the size of the window
        self.setFixedSize(512, 512)

        self.graph_widget = pg.GraphicsLayoutWidget()
        self.view = self.graph_widget.addViewBox()

        # Set some loop rates.
        self.viewer_freq = 30.0
        self.camera_freq = 15.0
        self.smoother_freq = 10.0

        # Image item
        self.image_item = pg.ImageItem(border="w")
        self.view.addItem(self.image_item)
        self.current_image = torch.zeros((3, 256, 256))
        self.current_keypoints = None

        # Scatter plot item
        self.scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120)
        )
        self.view.addItem(self.scatter)

        # Setup timer for GUI update.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.viewer_freq))

        self.setCentralWidget(self.graph_widget)

        self.detector = KeypointCNN()
        self.detector.load_state_dict(torch.load("outputs/models/m84lw6vs.pth"))
        self.detector.eval()
        # self.detector_compiled = torch.compile(self.detector)
        self.detector_compiled = self.detector

        self.init_smoother()

    def init_smoother(self):
        self.dynamics = partial(zero_acc_euler_dynamics, dt=1 / self.smoother_freq)
        info = self.zed_camera.camera.get_camera_information()
        camera_calibration = info.camera_configuration.calibration_parameters.left_cam
        fx, fy, cx, cy = (
            camera_calibration.fx,
            camera_calibration.fy,
            camera_calibration.cx,
            camera_calibration.cy,
        )

        MJC_CUBE_SCALE = 0.03  # 6cm cube on a side.
        self.object_frame_keypoints = torch.tensor(UNIT_CUBE_KEYPOINTS) * MJC_CUBE_SCALE

        keypoint_cfg = KeypointConfig(
            camera_intrinsics=torch.tensor([[fx, 0, 128], [0, fy, 128], [0, 0, 1]]),
            keypoints=self.object_frame_keypoints,
        )
        self.measurement = partial(keypoint_measurement, cfg=keypoint_cfg)

        smoother_config = SmootherConfig(
            horizon=15,
            init_pose_mean=pp.SE3([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]),
            # init_pose_covariance=torch.diag(
            #     torch.tensor([1e-1, 1e-1, 1e-1, 1.0, 1.0, 1.0])
            # ),
            init_pose_covariance=5e-2 * torch.eye(6),
            init_vel_covariance=1e-1 * torch.eye(6),
            Q_vel=torch.eye(6),
        )

        self.smoother = FixedLagSmoother(
            smoother_config,
            self.dynamics,
            self.measurement,
        )

    async def update_image(self):
        while True:
            try:
                start = time.time()
                frame = await self.zed_camera.get_frame_async()
                if frame is not None:
                    # Convert frame to the format suitable for pyqtgraph
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Run inference.
                    with torch.no_grad():
                        image, keypoints = self.get_detector_keypoints(frame)

                    # Update keypoints
                    self.current_keypoints = keypoints.clone()
                    self.current_image = image.clone()
                else:
                    raise ValueError("frame is None")

                end = time.time()
                print(f"Camera took {end - start} seconds to process.")

                await asyncio.sleep(1 / self.camera_freq)
            except Exception as e:
                print(f"An error occurred in update_image: {e}")

    async def update_smoother(self):
        while True:
            start = time.time()
            if self.current_keypoints is not None:
                self.smoother.update(self.current_keypoints)
                for _ in range(3):
                    self.smoother.step()
            end = time.time()

            print(f"Smoother took {end - start} seconds to process.")
            print(self.smoother.trajectory.poses)

            await asyncio.sleep(1 / self.smoother_freq)

    def update_frame(self):
        start = time.time()

        pixel_coordinates = (
            self.measurement(self.smoother.trajectory.poses[-1:]).squeeze().clone()
        )
        # pixel_coordinates = self.current_keypoints.clone()
        pixel_coordinates[:, 1] = self.detector.H - pixel_coordinates[:, 1]
        self.update_scatter(pixel_coordinates)

        # View cropped image
        self.image_item.setImage(
            cv2.flip(
                self.current_image.squeeze().permute(1, 2, 0).cpu().numpy(), 0
            ).transpose(1, 0, 2)
        )

        # # Force image to display at true size.
        # self.view.setRange(QtCore.QRectF(0, 0, self.detector.W, self.detector.H))

        end = time.time()
        print(f"Frame took {end - start} seconds to process.")

        # print(
        #     self.smoother.trajectory.poses[-1],
        #     self.smoother.trajectory.velocities[-1],
        # )

    def get_detector_keypoints(self, frame):
        # Convert to tensor
        image = kornia.utils.image_to_tensor(frame).unsqueeze(0) / 255.0

        # Scale image to detector size
        # image = kornia.geometry.transform.resize(image, self.detector.H)
        image = kornia.geometry.transform.center_crop(
            image, (self.detector.H, self.detector.W)
        )

        # Forward pass.
        net_start = time.time()
        raw_pixel_coordinates = self.detector_compiled(image).reshape(-1, 2).detach()
        net_end = time.time()
        print(f"Forward pass took {net_end - net_start} seconds.")
        predicted_pixel_coordinates = kornia.geometry.denormalize_pixel_coordinates(
            raw_pixel_coordinates, self.detector.H, self.detector.W
        ).cpu()

        # smaller_side = min(frame.shape[:2])
        # predicted_pixel_coordinates = kornia.geometry.denormalize_pixel_coordinates(
        #     raw_pixel_coordinates, smaller_side, smaller_side
        # ).cpu()

        # Convert center crop back to original size
        # width_diff = (frame.shape[1] - smaller_side) // 2
        # height_diff = (frame.shape[0] - smaller_side) // 2
        # predicted_pixel_coordinates[:, 0] += width_diff
        # predicted_pixel_coordinates[:, 1] += height_diff

        # predicted_pixel_coordinates[:, 1] = (
        #     frame.shape[0] - predicted_pixel_coordinates[:, 1]
        # )
        # print(predicted_pixel_coordinates)

        return image, predicted_pixel_coordinates

        # return np.array([[frame.shape[1] // 2, frame.shape[0] // 2]])

    async def run(self):
        self.camera_task = asyncio.create_task(self.update_image())
        self.smoother_task = asyncio.create_task(self.update_smoother())
        return self.camera_task, self.smoother_task

    def update_scatter(self, keypoints):
        scatter_data = [{"pos": kp, "size": 10} for kp in keypoints]
        self.scatter.setData(scatter_data)

    def closeEvent(self, event):
        self.zed_camera.close()
        super(MainWindow, self).closeEvent(event)


async def main():
    zed_camera = ZEDCamera()
    main_window = MainWindow(zed_camera)
    main_window.show()
    camera_task, smoother_task = await main_window.run()
    await asyncio.gather(camera_task, smoother_task)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    qasync.run(main())
    sys.exit(app.exec_())
