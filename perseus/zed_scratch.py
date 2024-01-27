from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer  # Import QTimer from PyQt5.QtCore
from pyzed import sl
import sys
import cv2
import numpy as np

camera = sl.Camera(2)
print(sl.Camera.get_device_list())
import sys

# Set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA  # Use HD1080 video mode
init_params.camera_fps = 30  # Set fps at 30

init_params.set_from_serial_number(33143189)

camera.open(init_params)


# Create a function to update the image
def update_image():
    # Capture an image from the Zed camera
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    if camera.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        camera.retrieve_image(image, sl.VIEW.LEFT)
        bgr_image = image.get_data()
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image to QImage and update the QLabel
        qt_image = QImage(
            rgb_image.data,
            rgb_image.shape[1],
            rgb_image.shape[0],
            rgb_image.strides[0],
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)


# Create a PyQt5 application
app = QApplication(sys.argv)

# Create a QMainWindow
window = QMainWindow()
window.setWindowTitle("Real-Time Image Display")

# Create a central widget to hold the QLabel
central_widget = QWidget()
window.setCentralWidget(central_widget)

# Create a QVBoxLayout for the central widget
layout = QVBoxLayout(central_widget)

# Create a QLabel to display the QPixmap
label = QLabel()
layout.addWidget(label)

# Create a timer to update the image at the desired frame rate (30 FPS)
timer = QTimer()
timer.timeout.connect(update_image)
timer.start(1000 // 30)  # Update every 1/30th of a second

# Show the main window
window.show()

# Start the PyQt5 event loop
sys.exit(app.exec_())
