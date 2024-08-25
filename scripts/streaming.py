import sys

import cv2
import kornia
import numpy as np
import torch
from pyzed import sl

from perseus import ROOT
from perseus.detector.models import KeypointCNN


class ZEDCamera:
    """A class for handling Zed camera I/O."""

    def __init__(self, serial_number: int, depth: bool = True, side: str = "left") -> None:
        """Initialize the ZED camera."""
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
        init_params.camera_image_flip = sl.FLIP_MODE.OFF
        init_params.camera_resolution = sl.RESOLUTION.VGA
        init_params.camera_fps = 100
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL if depth else sl.DEPTH_MODE.NONE
        init_params.depth_stabilization = 100
        init_params.coordinate_units = sl.UNIT.METER

        if depth:
            init_params.depth_minimum_distance = 0.1
            init_params.depth_maximum_distance = 0.5
            self.depth_buffer = sl.Mat()
        else:
            init_params.depth_minimum_distance = 0.3
            init_params.depth_maximum_distance = 1.0
        init_params.set_from_serial_number(serial_number)

        self.runtime_parameters.enable_depth = depth
        if depth:
            self.runtime_parameters.enable_fill_mode = True

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Didn't open!")
            sys.exit(1)

    def get_frame(self) -> np.ndarray | None:
        """Get a frame from the camera.

        Returns:
            np.ndarray: The frame as a numpy array. Shape: (256, 256, 3) or (256, 256, 4) if depth is enabled.
                None if the frame retrieval fails.
        """
        if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.rgb_buffer, self.rgb_view)
            frame = self.rgb_buffer.get_data()[..., :3]  # (H, W, C), bgr, uint8
            frame = frame[..., ::-1] / 255.0  # Convert to RGB, divide by 255 to normalize
            assert frame.shape[2] == 3, "Frame should have 3 channels."  # noqa: PLR2004
            if self.depth:
                self.camera.retrieve_measure(self.depth_buffer, self.depth_measure)
                depth = self.depth_buffer.get_data()
                depth[np.isnan(depth)] = 0
                depth[np.isinf(depth)] = 0
                depth /= 0.035  # unscale to the scale of the training data
                frame = np.concatenate([frame, depth[..., None]], axis=-1)

            H, W = frame.shape[:2]
            frame = frame[H // 2 - 128 : H // 2 + 128, W // 2 - 128 : W // 2 + 128, ...]
            return frame
        return None

    def close(self) -> None:
        """Close the camera."""
        self.camera.close()


def main(serial_number: int, window_size: int = 800) -> None:
    """Main function for streaming ZED camera feed and displaying keypoint predictions.

    Args:
        serial_number: The serial number of the ZED camera.
        window_size: The side length of the square window in pixels.
    """
    window_name = "ZED Camera Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_size, window_size)  # Adjust the size as needed

    zed_camera = ZEDCamera(serial_number)  # options: 12746523, 14390641, 19798856
    model = KeypointCNN(num_channels=4 if zed_camera.depth else 3)
    state_dict = torch.load(f"{ROOT}/outputs/models/wzbx1og6.pth", weights_only=True)
    for key in list(state_dict.keys()):
        if "module." in key:
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()

    while True:
        frame = zed_camera.get_frame()
        if frame is None:
            continue

        with torch.no_grad():
            image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
            raw_pixel_coordinates = model(image_tensor[None, ...]).reshape(-1, 2).detach()
            keypoints = (
                kornia.geometry.denormalize_pixel_coordinates(raw_pixel_coordinates, model.H, model.W).cpu().numpy()
            )

        # Draw keypoints
        for kp in keypoints:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), radius=5, color=(0, 0, 1), thickness=-1)  # Red keypoints

        resized_frame = cv2.resize((frame[..., :3][..., ::-1] * 255).astype(np.uint8), (window_size, window_size))
        cv2.imshow(window_name, resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    zed_camera.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    serial_number = 12746523  # options: 12746523, 14390641, 19798856
    window_size = 800  # side length of square window in pixels
    main(serial_number, window_size=window_size)
