# Copyright 2023 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import shutil

import bpy
import numpy as np
from custom_renderer import CustomBlender as Blender
from local_asset_source import LocalAssetSource

import kubric as kb
from kubric.simulator import PyBullet
from perseus import ROOT

# --- Some configuration values
# the region in which to place objects [(min), (max)]
STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 10)]
DYNAMIC_SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
MJC_SPAWN_REGION = [(-4.0, -4.0, 1.0), (4.0, 4.0, 0.0)]
VELOCITY_RANGE = [(-4.0, -4.0, 0.0), (4.0, 4.0, 0.0)]
MJC_VELOCITY_RANGE = [(-1.0, -1.0, -0.5), (1.0, 1.0, 1.0)]
MJC_ANGULAR_VELOCITY_RANGE = [(-4.0, -4.0, -4.0), (4.0, 4.0, 4.0)]

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"], default="train")
# Configuration for the objects of the scene
parser.add_argument(
    "--min_num_static_objects",
    type=int,
    default=10,
    help="minimum number of static (distractor) objects",
)
parser.add_argument(
    "--max_num_static_objects",
    type=int,
    default=20,
    help="maximum number of static (distractor) objects",
)
parser.add_argument(
    "--min_num_dynamic_objects",
    type=int,
    default=1,
    help="minimum number of dynamic (tossed) objects",
)
parser.add_argument(
    "--max_num_dynamic_objects",
    type=int,
    default=3,
    help="maximum number of dynamic (tossed) objects",
)
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement"], default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=4.0)
parser.add_argument("--max_motion_blur", type=float, default=0.5)


# Configuration for the source of the assets
parser.add_argument(
    "--kubasic_assets",
    type=str,
    default=f"{ROOT}/data_generation/assets/KuBasic.json",
)
parser.add_argument(
    "--hdri_assets",
    type=str,
    default=f"{ROOT}/data_generation/assets/HDRI_haven.json",
)
parser.add_argument("--gso_assets", type=str, default=f"{ROOT}/data_generation/assets/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12, resolution=256)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
output_dir = ROOT / output_dir  # changing the output_dir to an absolute path

try:
    motion_blur = rng.uniform(0, FLAGS.max_motion_blur)
    # motion_blur = 0.0
    if motion_blur > 0.0:
        logging.info(f"Using motion blur strength {motion_blur}")

    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(
        scene,
        scratch_dir,
        use_denoising=True,
        samples_per_pixel=64,
        motion_blur=motion_blur,
    )

    # Setup some other stuff for performance.
    renderer.blender_scene.render.use_persistent_data = True
    logging.info("use_gpu: %s", renderer.use_gpu)
    logging.info("Blender device: %s", renderer.blender_scene.cycles.device)

    kubasic = LocalAssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = LocalAssetSource.from_manifest(FLAGS.gso_assets)
    hdri_source = LocalAssetSource.from_manifest(FLAGS.hdri_assets)

    # --- Populate the scene
    # background HDRI
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    if FLAGS.backgrounds_split == "train":
        logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
        hdri_id = rng.choice(train_backgrounds)
    else:
        logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
        hdri_id = rng.choice(test_backgrounds)
    background_hdri = hdri_source.create(asset_id=hdri_id)
    # assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", hdri_id)
    scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(
        asset_id="dome",
        name="dome",
        friction=1.0,
        restitution=0.0,
        static=True,
        background=True,
    )
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    def get_linear_camera_motion_start_end(
        movement_speed: float,
        inner_radius: float = 8.0,
        outer_radius: float = 12.0,
        z_offset: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a linear path which starts and ends within a half-sphere shell."""
        while True:
            camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius, outer_radius, z_offset))
            direction = rng.rand(3) - 0.5
            movement = direction / np.linalg.norm(direction) * movement_speed
            camera_end = camera_start + movement
            if inner_radius <= np.linalg.norm(camera_end) <= outer_radius and camera_end[2] > z_offset:
                return camera_start, camera_end

    def get_linear_lookat_motion_start_end(
        inner_radius: float = 1.0,
        outer_radius: float = 4.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a linear path which goes through the workspace center."""
        while True:
            # Sample a point near the workspace center that the path travels through
            camera_through = np.array(kb.sample_point_in_half_sphere_shell(0.0, inner_radius, 0.0))
            while True:
                # Sample one endpoint of the trajectory
                camera_start = np.array(kb.sample_point_in_half_sphere_shell(0.0, outer_radius, 0.0))
                if camera_start[-1] < inner_radius:
                    break

            # Continue the trajectory beyond the point in the workspace center, so the
            # final path passes through that point.
            continuation = rng.rand(1) * 0.5
            camera_end = camera_through + continuation * (camera_through - camera_start)

            # Second point will probably be closer to the workspace center than the
            # first point.  Get extra augmentation by randomly swapping first and last.
            if rng.rand(1)[0] < 0.5:  # noqa: PLR2004
                tmp = camera_start
                camera_start = camera_end
                camera_end = tmp
            return camera_start, camera_end

    # Camera
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=35.0, sensor_width=32)
    if FLAGS.camera == "fixed_random":
        scene.camera.position = kb.sample_point_in_half_sphere_shell(inner_radius=7.0, outer_radius=9.0, offset=0.2)
        scene.camera.look_at((0, 0, 0))
    elif FLAGS.camera in ["linear_movement", "linear_movement_linear_lookat"]:
        is_panning = FLAGS.camera == "linear_movement_linear_lookat"
        camera_inner_radius = 6.0 if is_panning else 8.0
        camera_start, camera_end = get_linear_camera_motion_start_end(
            movement_speed=rng.uniform(low=0.0, high=FLAGS.max_camera_movement)
        )
        if is_panning:
            lookat_start, lookat_end = get_linear_lookat_motion_start_end()

        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames
        for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
            interp = (frame - FLAGS.frame_start + 1) / (FLAGS.frame_end - FLAGS.frame_start + 3)
            scene.camera.position = interp * np.array(camera_start) + (1 - interp) * np.array(camera_end)
            if is_panning:
                scene.camera.look_at(interp * np.array(lookat_start) + (1 - interp) * np.array(lookat_end))
            else:
                scene.camera.look_at((0, 0, 0))
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)

    # ---- Object placement ----
    train_split, test_split = gso.get_test_split(fraction=0.1)
    if FLAGS.objects_split == "train":
        logging.info("Choosing one of the %d training objects...", len(train_split))
        active_split = train_split
    else:
        logging.info("Choosing one of the %d held-out objects...", len(test_split))
        active_split = test_split

    # add STATIC objects
    num_static_objects = rng.randint(FLAGS.min_num_static_objects, FLAGS.max_num_static_objects + 1)
    logging.info("Randomly placing %d static objects:", num_static_objects)
    for _ in range(num_static_objects):
        obj = gso.create(asset_id=rng.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng.uniform(0.75, 3.0)
        abs_scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.scale = abs_scale
        obj.metadata["scale"] = scale
        obj.metadata["abs_scale"] = abs_scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=STATIC_SPAWN_REGION, rng=rng)
        obj.friction = 1.0
        obj.restitution = 0.0
        obj.metadata["is_dynamic"] = False
        obj.metadata["render_filename"] = obj.render_filename
        logging.info("    Added %s at %s", obj.asset_id, obj.position)

    logging.info("Running 100 frames of simulation to let static objects settle ...")
    _, _ = simulator.run(frame_start=-100, frame_end=0)

    # stop any objects that are still moving and reset friction / restitution
    for obj in scene.foreground_assets:
        if hasattr(obj, "velocity"):
            obj.velocity = (0.0, 0.0, 0.0)
            obj.friction = 0.5
            obj.restitution = 0.5

    dome.friction = FLAGS.floor_friction
    dome.restitution = FLAGS.floor_restitution

    mjc = kb.FileBasedObject(
        asset_id="mjc",
        render_filename=f"{ROOT}/data_generation/assets/mjc.glb",
        bounds=((-1, -1, -1), (1, 1, 1)),
        simulation_filename=f"{ROOT}/data_generation/assets/mjc.urdf",
    )
    mjc.velocity = rng.uniform(*MJC_VELOCITY_RANGE) - [
        mjc.position[0],
        mjc.position[1],
        0,
    ]
    mjc.angular_velocity = rng.uniform(*MJC_ANGULAR_VELOCITY_RANGE)
    scale = rng.uniform(0.75, 3.0)
    abs_scale = scale / np.max(mjc.bounds[1] - mjc.bounds[0])
    mjc.scale = abs_scale
    mjc.metadata["scale"] = scale
    mjc.metadata["abs_scale"] = abs_scale
    mjc.metadata["is_dynamic"] = True
    mjc.metadata["render_filename"] = mjc.render_filename
    scene += mjc

    # Randomize cube material properties.
    mjc_blender = mjc.linked_objects[renderer]
    roughness = rng.uniform(0.0, 0.3)
    specular = rng.uniform(0.75, 1.0)
    metallic = rng.uniform(0.25, 0.75)
    ior = rng.uniform(1.0, 2.0)
    for material in mjc_blender.data.materials:
        for node in material.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                node.inputs["Roughness"].default_value = roughness
                node.inputs["Specular"].default_value = specular
                node.inputs["Metallic"].default_value = metallic
                node.inputs["IOR"].default_value = ior
                print(specular, roughness)

    kb.move_until_no_overlap(mjc, simulator, spawn_region=MJC_SPAWN_REGION, rng=rng, max_trials=1000)

    # Add DYNAMIC objects
    num_dynamic_objects = rng.randint(FLAGS.min_num_dynamic_objects, FLAGS.max_num_dynamic_objects + 1)
    logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
    for _ in range(num_dynamic_objects):
        obj = gso.create(asset_id=rng.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        abs_scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.scale = abs_scale
        obj.metadata["scale"] = scale
        obj.metadata["abs_scale"] = abs_scale
        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=DYNAMIC_SPAWN_REGION, rng=rng)
        obj.velocity = rng.uniform(*VELOCITY_RANGE) - [
            obj.position[0],
            obj.position[1],
            0,
        ]
        obj.metadata["is_dynamic"] = True
        logging.info("    Added %s at %s", obj.asset_id, obj.position)
        obj.metadata["render_filename"] = obj.render_filename

    if FLAGS.save_state:
        logging.info(
            "Saving the simulator state to '%s' prior to the simulation.",
            output_dir / "scene.bullet",
        )
        simulator.save_state(output_dir / "scene.bullet")

    # Run dynamic objects simulation
    logging.info("Running the simulation ...")
    animation, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)

    # --- Rendering
    if FLAGS.save_state:
        logging.info("Saving the renderer state to '%s' ", output_dir / "scene.blend")
        renderer.save_state(output_dir / "scene.blend")

    logging.info("Rendering the scene ...")
    data_stack = renderer.render(return_layers=("rgba", "depth", "segmentation"))

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True,
    )

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"], scene.assets, visible_foreground_assets
    )
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # Save to image files
    kb.write_image_dict(data_stack, output_dir)
    kb.post_processing.compute_bboxes(data_stack["segmentation"], visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(
        filename=output_dir / "metadata.json",
        data={
            "flags": vars(FLAGS),
            "metadata": kb.get_scene_metadata(scene),
            "camera": kb.get_camera_info(scene.camera),
            "instances": kb.get_instance_info(scene, visible_foreground_assets),
        },
    )
    kb.write_json(
        filename=output_dir / "events.json",
        data={
            "collisions": kb.process_collisions(collisions, scene, assets_subset=visible_foreground_assets),
        },
    )

    kb.done()

except Exception as e:
    logging.exception("An error occurred during scene generation.")
    shutil.rmtree(output_dir)
    shutil.rmtree(scratch_dir)

    raise e

    kb.done()
