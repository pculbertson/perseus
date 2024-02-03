import sys
import time

import pyzed.sl as sl

import torch
import kornia
import numpy as np

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import asyncio
import qasync
import cv2

import gtsam
from perseus.smoother.factors import (
    PoseDynamicsFactor,
    ConstantVelocityFactor,
    KeypointProjectionFactor,
)
from gtsam.symbol_shorthand import X, V, W
import gtsam_unstable

from perseus.detector.models import KeypointCNN
from perseus.smoother.utils import UNIT_CUBE_KEYPOINTS


class ZEDCamera:
    """
    A class for handling Zed camera I/O.
    """

    def __init__(self):
        """
        Initializes the Zed camera. Currently hard-coded for 1080 resolution
        and a particular serial number.
        """
        # TODO: Expose parameters for camera initialization.

        # Create containers for camera/image.
        self.camera = sl.Camera()
        self.image = (
            sl.Mat()
        )  # This needs to be stored or the image will be garbage collected.
        self.runtime_parameters = sl.RuntimeParameters()

        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        init_params.set_from_serial_number(33143189)

        # Open the camera
        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

    async def get_frame_async(self):
        """
        An async wrapper around `get_frame` that allows for non-blocking
        camera I/O.
        """
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self.get_frame)
        return frame

    def get_frame(self):
        """
        Literally ChatGPT boilerplate for getting a frame from the Zed camera.
        """
        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
            frame = self.image.get_data()
            return frame

        return None  # Return None if no frame is available.

    def close(self):
        # Close the ZED camera
        self.camera.close()


class MainWindow(QtWidgets.QMainWindow):
    """
    A PyQT app for running the Zed camera and state estimator.
    """

    def __init__(self, zed_camera, parent=None):
        """
        Intializes the app and constructs all needed objects.
        """
        # TODO: Expose config parameters (possibly all the way to command line).
        super(MainWindow, self).__init__(parent)

        self.zed_camera = zed_camera

        # Fix the size of the window and create layout containers.
        self.setFixedSize(512, 512)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.view = self.graph_widget.addViewBox()
        self.setCentralWidget(self.graph_widget)

        # Set some loop rates.
        self.viewer_freq = 30.0
        self.camera_freq = 30.0
        self.smoother_freq = 30.0

        # Setup image item.
        self.image_item = pg.ImageItem(border="w")
        self.view.addItem(self.image_item)
        self.current_image = torch.zeros((3, 256, 256))
        self.current_keypoints = None

        # Setup scatter plot for estimated keypoints.
        self.scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120)
        )
        self.view.addItem(self.scatter)

        # Setup QT timer for GUI update -- other loops handled via asyncio.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.viewer_freq))

        # Load the detector and model weights.
        self.detector = KeypointCNN()
        self.detector.load_state_dict(torch.load("outputs/models/m84lw6vs.pth"))
        self.detector.eval()

        # Set up the factor graph.
        self.init_smoother()

    def init_smoother(self):
        """
        Sets up all data needed for the smoother (priors, camera cal, etc.)
        and intiializes the factor graph.
        """

        # Create camera calibration by reading from Zed.
        info = self.zed_camera.camera.get_camera_information()
        camera_calibration = info.camera_configuration.calibration_parameters.left_cam
        fx, fy, cx, cy = (
            camera_calibration.fx,
            camera_calibration.fy,
            camera_calibration.cx,
            camera_calibration.cy,
        )
        s = 0.0
        self.calibration = gtsam.Cal3_S2(fx, fy, s, cx, cy)

        # Load and scale keypoints.
        MJC_CUBE_SCALE = 0.03  # 6cm cube on a side.
        self.object_frame_keypoints = torch.tensor(UNIT_CUBE_KEYPOINTS) * MJC_CUBE_SCALE

        # Setup the smoother.
        HORIZON = 5  # Number of frames to look back.
        lag = HORIZON * 1 / self.smoother_freq  # Number of seconds to look back.

        # Create a GTSAM fixed-lag smoother.
        self.smoother = gtsam_unstable.BatchFixedLagSmoother(lag)
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        self.new_timestamps = gtsam_unstable.FixedLagSmootherKeyTimestampMap()

        # Create noise model for the keypoint projection factor.
        self.keypoint_noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            5 * np.array([1e0, 1e0])
        )

        # Define prior parameters for the pose and velocity.
        prior_pose_mean = gtsam.Pose3(gtsam.Rot3(), np.array([0.0, 0.0, 1.25]))
        prior_pose_cov = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e0, 1e0, 1e0, 1e0, 1e0, 1e0])
        )
        prior_vel_mean = np.array([0.0, 0.0, 0.0])
        prior_vel_cov = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.25, 0.25, 0.25]))
        prior_ang_vel_mean = np.array([0.0, 0.0, 0.0])
        prior_ang_vel_cov = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))

        # Create process noise models.
        self.Q_pose = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])
        )
        self.Q_vel = prior_vel_cov
        self.Q_ang_vel = prior_ang_vel_cov

        # Add the priors to the graph.
        self.new_factors.push_back(
            gtsam.PriorFactorPose3(X(0), prior_pose_mean, prior_pose_cov)
        )
        self.new_factors.push_back(
            gtsam.PriorFactorVector(V(0), prior_vel_mean, prior_vel_cov)
        )
        self.new_factors.push_back(
            gtsam.PriorFactorVector(W(0), prior_ang_vel_mean, prior_ang_vel_cov)
        )

        # Add the initial values to the graph.
        self.new_values.insert(X(0), prior_pose_mean)
        self.new_values.insert(V(0), prior_vel_mean)
        self.new_values.insert(W(0), prior_ang_vel_mean)

        # Init counters we'll use for the dt/iteration.
        self.last_smoother_time = time.time()
        self.smoother_iter = 0
        self.last_image_time = None

        # Initialize camera object we can use to project points into image frame.
        self.gtsam_camera = gtsam.PinholeCameraCal3_S2(gtsam.Pose3(), self.calibration)

        # Add the initial timestamp.
        self.new_timestamps.insert((X(0), self.last_smoother_time))
        self.new_timestamps.insert((V(0), self.last_smoother_time))
        self.new_timestamps.insert((W(0), self.last_smoother_time))

        # Update the graph and store results to get started.
        self.smoother.update(self.new_factors, self.new_values, self.new_timestamps)
        self.result = self.smoother.calculateEstimate()
        self.new_timestamps.clear()
        self.new_factors.resize(0)
        self.new_values.clear()

    async def update_image(self):
        """
        Async function for reading the Zed camera and running inference.
        """
        while True:
            try:
                start = time.time()
                frame = await self.zed_camera.get_frame_async()  # Read camera.
                if frame is not None:
                    # Store timing from frame.
                    self.last_image_time = time.time()

                    # Convert frame to the format suitable for pyqtgraph
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Run model inference to get est. keypoints.
                    with torch.no_grad():
                        image, keypoints = self.get_detector_keypoints(frame)

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
            if self.last_image_time is None or self.current_keypoints is None:
                # Skip if we don't have a new image yet.
                pass
            elif self.last_image_time > self.last_smoother_time:
                # If we have a new image, update the smoother.

                start = time.time()
                self.smoother_iter += 1

                # Compute current dt.
                dt = self.last_image_time - self.last_smoother_time

                # Store the current time.
                self.last_smoother_time = self.last_image_time

                # Add new timestamps to the graph.
                self.new_timestamps.insert(
                    (X(self.smoother_iter), self.last_image_time)
                )
                self.new_timestamps.insert(
                    (V(self.smoother_iter), self.last_image_time)
                )
                self.new_timestamps.insert(
                    (W(self.smoother_iter), self.last_image_time)
                )

                # Create keypoint factors and add them to the graph.
                for i, keypoint in enumerate(self.current_keypoints):
                    keypoint_measurement = keypoint.numpy()
                    point_body_frame = self.object_frame_keypoints[i].numpy()
                    self.new_factors.push_back(
                        KeypointProjectionFactor(
                            X(self.smoother_iter),
                            self.keypoint_noise_model,
                            self.calibration,
                            keypoint_measurement,
                            point_body_frame,
                        )
                    )

                # Add a pose dynamics factor to the graph.
                self.new_factors.push_back(
                    PoseDynamicsFactor(
                        X(self.smoother_iter - 1),
                        W(self.smoother_iter - 1),
                        V(self.smoother_iter - 1),
                        X(self.smoother_iter),
                        self.Q_pose,
                        dt,
                    )
                )

                # Add constant velocity factor to the graph.
                self.new_factors.push_back(
                    ConstantVelocityFactor(
                        V(self.smoother_iter - 1), V(self.smoother_iter), self.Q_vel
                    )
                )

                # Add constant angular velocity factor to the graph.
                self.new_factors.push_back(
                    ConstantVelocityFactor(
                        W(self.smoother_iter - 1),
                        W(self.smoother_iter),
                        self.Q_ang_vel,
                    )
                )

                # Update initial values.
                self.new_values.insert(
                    X(self.smoother_iter),
                    self.result.atPose3(X(self.smoother_iter - 1)),
                )
                self.new_values.insert(
                    V(self.smoother_iter),
                    self.result.atVector(V(self.smoother_iter - 1)),
                )
                self.new_values.insert(
                    W(self.smoother_iter),
                    self.result.atVector(W(self.smoother_iter - 1)),
                )

                # Update the graph and store results.
                self.smoother.update(
                    self.new_factors, self.new_values, self.new_timestamps
                )
                self.result = self.smoother.calculateEstimate()
                self.new_timestamps.clear()
                self.new_factors.resize(0)
                self.new_values.clear()

                print(f"Smoother update took {time.time() - start} seconds.")

            await asyncio.sleep(1 / self.smoother_freq)

    def update_frame(self):
        """
        Draws the current image and keypoints to the screen.
        """
        start = time.time()

        # Transform body-frame keypoints to camera frame + project to pixel coordinates.
        pixel_coordinates = np.array(
            [
                self.gtsam_camera.project(
                    self.result.atPose3(X(self.smoother_iter)).transformFrom(kk)
                )
                for kk in self.object_frame_keypoints.numpy()
            ]
        )

        # For viz: flip y-axis to match PyQT format.
        pixel_coordinates[:, 1] = self.detector.H - pixel_coordinates[:, 1]
        self.update_scatter(pixel_coordinates)

        # View cropped image -- also flip for PyQT.
        self.image_item.setImage(
            cv2.flip(
                self.current_image.squeeze().permute(1, 2, 0).cpu().numpy(), 0
            ).transpose(1, 0, 2)
        )

        end = time.time()
        print(f"Frame took {end - start} seconds to process.")

    def get_detector_keypoints(self, frame):
        """
        Util to run the detector and convert the output to pixel coordinates.
        """
        # Convert image to tensor.
        image = kornia.utils.image_to_tensor(frame).unsqueeze(0) / 255.0

        # Center crop the hi-res image to the detector's input size.
        image = kornia.geometry.transform.center_crop(
            image, (self.detector.H, self.detector.W)
        )

        # Pass image through CNN.
        net_start = time.time()
        raw_pixel_coordinates = self.detector(image).reshape(-1, 2).detach()
        net_end = time.time()
        print(f"Forward pass took {net_end - net_start} seconds.")
        predicted_pixel_coordinates = kornia.geometry.denormalize_pixel_coordinates(
            raw_pixel_coordinates, self.detector.H, self.detector.W
        ).cpu()

        return image, predicted_pixel_coordinates

    async def run(self):
        """
        Sets up the async tasks and runs the main loop.
        """
        self.camera_task = asyncio.create_task(self.update_image())
        self.smoother_task = asyncio.create_task(self.update_smoother())
        return self.camera_task, self.smoother_task

    def update_scatter(self, keypoints):
        """
        Util to update scatter plot with new keypoints.
        """
        scatter_data = [{"pos": kp, "size": 10} for kp in keypoints]
        self.scatter.setData(scatter_data)

    def closeEvent(self, event):
        """
        Shutdown the Zed camera when the app is closed.
        """
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
