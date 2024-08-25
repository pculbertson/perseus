import asyncio
import sys
import time

import cv2
import kornia
import numpy as np
import pyqtgraph as pg
import qasync
import torch
from PyQt5 import QtCore, QtWidgets
from pyzed import sl

from perseus import ROOT
from perseus.detector.models import KeypointCNN

"""Useful values.

ZED2i camera serial numbers:
    * cam A: 33143189
    * cam B: 32144978
    * cam C: 35094665

ZED Mini camera serial numbers:
    * cam A: 12746523
    * cam B: 14390641
    * cam C: 19798856
"""


class ZEDCamera:
    """A class for handling Zed camera I/O."""

    def __init__(self, serial_number: int, depth: bool = True, side: str = "left") -> None:
        """Initializes a ZED camera.

        Assumes VGA resolution at 100 FPS, which is what we run in the ROS stack.

        Args:
            serial_number: The serial number of the ZED camera.
            depth: Whether to enable depth sensing.
            side: The side of the camera to use. Either "left" or "right".
        """
        self.depth = depth
        if side == "left":
            self.rgb_view = sl.VIEW.LEFT
            if self.depth:
                self.depth_measure = sl.MEASURE.DEPTH
        else:
            self.rgb_view = sl.VIEW.RIGHT
            if self.depth:
                self.depth_measure = sl.MEASURE.DEPTH_RIGHT

        self.camera = sl.Camera()
        self.rgb_buffer = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()

        # Set initialization parameters
        init_params = sl.InitParameters()
        init_params.camera_image_flip = sl.FLIP_MODE.OFF  # don't automatically flip the camera based on orientation
        init_params.camera_resolution = sl.RESOLUTION.VGA  # Use VGA video mode
        init_params.camera_fps = 100  # Set FPS to 100
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL if depth else sl.DEPTH_MODE.NONE
        init_params.depth_stabilization = 100
        init_params.coordinate_units = sl.UNIT.METER

        # if depth, we assume we're using a ZED Mini, whose min distance is 0.1. ZED2i is 0.3.
        if depth:
            init_params.depth_minimum_distance = 0.1
            init_params.depth_maximum_distance = 0.5
            self.depth_buffer = sl.Mat()  # need a buffer for depth images
        else:
            init_params.depth_minimum_distance = 0.3
            init_params.depth_maximum_distance = 1.0
        init_params.set_from_serial_number(serial_number)  # Set the serial number of the camera

        # set runtime parameters
        self.runtime_parameters.enable_depth = depth
        if depth:
            self.runtime_parameters.enable_fill_mode = True

        # Open the camera
        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Didn't open!")
            sys.exit(1)

    def get_frame(self) -> np.ndarray | None:
        """Retrieves a frame from the camera.

        Returns:
            The frame as a numpy array, or None if no frame is available.
        """
        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.image, self.rgb_view)
            frame = self.image.get_data()[..., :3]  # (H, W, C), bgr
            frame = frame[..., ::-1]  # Convert to RGB
            assert frame.shape[2] == 3, "Frame should have 3 channels."  # noqa: PLR2004
            if self.depth:
                self.camera.retrieve_measure(self.depth_buffer, self.depth_measure)
                depth = self.depth_buffer.get_data()
                frame = np.concatenate([frame, depth[..., None]], axis=-1)

            # center crop to 256 x 256
            H, W = frame.shape[:2]
            frame = frame[H // 2 - 128 : H // 2 + 128, W // 2 - 128 : W // 2 + 128, ...]
            return frame

        return None  # Return None if no frame is available.

    async def get_frame_async(self) -> np.ndarray:
        """An async wrapper around `get_frame` that allows for non-blocking camera I/O."""
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(None, self.get_frame)
        return frame

    def close(self) -> None:
        """Closes the camera."""
        self.camera.close()


class MainWindow(QtWidgets.QMainWindow):
    """A PyQT app for running the Zed camera and keypoint model in real time."""

    def __init__(
        self,
        zed_camera: ZEDCamera,
        ckpt_path: str = f"{ROOT}/outputs/models/wzbx1og6.pth",  # RGBD pruned model
        dataset_path: str = f"{ROOT}/data/pruned_dataset/pruned.hdf5",  # RGBD pruned dataset
        pruned: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Intializes the app.

        Args:
            zed_camera: The ZED camera object.
            ckpt_path: The path to the model checkpoint.
            dataset_path: The path to the dataset.
            pruned: Whether the dataset is pruned or not.
            parent: The parent widget.
        """
        super().__init__(parent)

        self.zed_camera = zed_camera

        # Fix the size of the window and create layout containers.
        self.setFixedSize(512, 512)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.view = self.graph_widget.addViewBox()
        self.setCentralWidget(self.graph_widget)

        # Set some loop rates.
        self.viewer_freq = 30.0
        self.camera_freq = zed_camera.camera.get_camera_fps()

        # Setup image item.
        self.image_item = pg.ImageItem(border="w")
        self.view.addItem(self.image_item)
        self.current_image = torch.zeros((3, 256, 256))
        self.current_keypoints = None

        # Setup scatter plot for estimated keypoints.
        self.scatter = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.view.addItem(self.scatter)

        # Setup QT timer for GUI update -- other loops handled via asyncio.
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.viewer_freq))

        # Load the detector and model weights.
        if zed_camera.depth:
            self.model = KeypointCNN(num_channels=4)
        else:
            self.model = KeypointCNN()
        state_dict = torch.load(ckpt_path, weights_only=True)
        for key in list(state_dict.keys()):
            if "module." in key:
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_detector_keypoints(self, frame: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Util to run the detector and convert the output to pixel coordinates.

        Args:
            frame: The input frame.

        Returns:
            image: The input image as a tensor.
            predicted_pixel_coordinates: The predicted pixel coordinates.
        """
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        raw_pixel_coordinates = self.model(image).reshape(-1, 2).detach()
        predicted_pixel_coordinates = kornia.geometry.denormalize_pixel_coordinates(
            raw_pixel_coordinates, self.model.H, self.model.W
        ).cpu()
        return image, predicted_pixel_coordinates

    async def update_image(self) -> None:
        """Async function for reading the Zed camera and running inference."""
        while True:
            try:
                frame = await self.zed_camera.get_frame_async()
                if frame is not None:
                    self.last_image_time = time.time()
                    with torch.no_grad():
                        image, keypoints = self.get_detector_keypoints(frame)
                    self.current_keypoints = keypoints.clone()
                    self.current_image = image.clone()
                else:
                    raise ValueError("frame is None")
                # print(f"Camera took {end - start} seconds to process.")
                await asyncio.sleep(1 / self.camera_freq)
            except Exception as e:
                print(f"An error occurred in update_image: {e}")

    def update_frame(self) -> None:
        """Draws the current image and keypoints to the screen."""
        pixel_coordinates = self.current_keypoints

        # For viz: flip y-axis to match PyQT format.
        pixel_coordinates[:, 1] = self.model.H - pixel_coordinates[:, 1]
        self.update_scatter(pixel_coordinates)

        # View cropped image -- also flip for PyQT.
        self.image_item.setImage(
            cv2.flip(self.current_image[..., :3].permute(1, 2, 0).cpu().numpy(), 0).transpose(1, 0, 2)
        )

    async def run(self) -> asyncio.Task:
        """Sets up the async tasks and runs the main loop."""
        self.camera_task = asyncio.create_task(self.update_image())
        return self.camera_task

    def update_scatter(self, keypoints: np.ndarray) -> None:
        """Util to update scatter plot with new keypoints."""
        brushes = [
            pg.mkBrush(255, 255, 255, 120) if i != 7 else pg.mkBrush(255, 0, 0, 120)  # noqa: PLR2004
            for i in range(8)
        ]
        scatter_data = [{"pos": kp, "size": 10} for kp in keypoints]
        self.scatter.setData(scatter_data)
        self.scatter.setBrush(brushes)

    def closeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        """Shutdown the Zed camera when the app is closed."""
        self.zed_camera.close()
        super(MainWindow, self).closeEvent(event)


async def main() -> None:
    """Main function for running the app."""
    zed_camera = ZEDCamera(12746523)  # zed mini serial numbers: 12746523, 14390641, 19798856
    main_window = MainWindow(zed_camera)
    main_window.show()
    camera_task = await main_window.run()
    await asyncio.gather(camera_task)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    qasync.run(main())
    sys.exit(app.exec_())
