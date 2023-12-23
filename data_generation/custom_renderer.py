from kubric.renderer import Blender
from kubric.safeimport.bpy import bpy
import logging

logger = logging.getLogger(__name__)


class CustomBlender(Blender):
    """
    Quick variant of kb.renderer.Blender that has CUDA suppport.
    """

    @property
    def use_gpu(self) -> bool:
        return self.blender_scene.cycles.device == "GPU"

    @use_gpu.setter
    def use_gpu(self, value: bool):
        self.blender_scene.cycles.device = "GPU" if value else "CPU"
        if value:
            # call get_devices() to let Blender detect GPU devices
            preferences = bpy.context.preferences
            cycles_preferences = preferences.addons["cycles"].preferences
            cycles_preferences.get_devices()
            cycles_preferences.compute_device_type = "CUDA"

            for device in cycles_preferences.devices:
                logger.info("Activating: %s", device.name)
                device.use = True
