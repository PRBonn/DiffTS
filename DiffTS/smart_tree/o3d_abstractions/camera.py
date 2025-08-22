import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


def create_camera(width, height, fx=575, fy=575):
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.extrinsic = np.eye(4)
    camera_parameters.intrinsic.set_intrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=(width / 2.0 - 0.5),
        cy=(height / 2.0 - 0.5),
    )
    return camera_parameters


def update_camera_position(
    camera, camera_position, camera_target=[0, 0, 0], up=np.asarray([0, 1, 0])
):
    camera_direction = (camera_target - camera_position) / np.linalg.norm(
        camera_target - camera_position
    )
    camera_right = np.cross(camera_direction, up)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    position_matrix = np.eye(4)
    position_matrix[:3, 3] = -camera_position
    camera_look_at = np.eye(4)
    camera_look_at[:3, :3] = np.vstack((camera_right, camera_up, camera_direction))
    camera_look_at = np.matmul(camera_look_at, position_matrix)
    camera.extrinsic = camera_look_at
    return camera


def o3d_headless_render(geoms, camera_position, camera_up):
    width, height = 1920, 1080
    camera = create_camera(width, height)

    # Setup a Offscreen Renderer
    render = rendering.OffscreenRenderer(width=width, height=height)
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA
    render.setup_camera(camera.intrinsic, camera.extrinsic)
    render.scene.scene.set_sun_light(
        geoms[0].get_center() + np.asarray(camera_position), [1.0, 1.0, 1.0], 75000
    )
    render.scene.scene.enable_sun_light(True)
    render.scene.scene.enable_indirect_light(True)
    render.scene.scene.set_indirect_light_intensity(0.3)
    # render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.shader = "defaultUnlit"

    for i, item in enumerate(geoms):
        render.scene.add_geometry(f"{i}", item, mtl)

    camera = update_camera_position(
        camera,
        geoms[0].get_center() + np.asarray(camera_position),
        camera_target=geoms[0].get_center(),
        up=np.asarray(camera_up),
    )
    render.setup_camera(camera.intrinsic, camera.extrinsic)

    return render.render_to_image()


class Renderer:
    def __init__(self, width, height):
        self.camera = create_camera(width, height)
        self.render = rendering.OffscreenRenderer(width=width, height=height)
        self.render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # RGBA
        self.render.setup_camera(self.camera.intrinsic, self.camera.extrinsic)

        self.render.scene.scene.enable_sun_light(True)
        self.render.scene.scene.enable_indirect_light(True)
        self.render.scene.scene.set_indirect_light_intensity(0.3)
        self.mtl = o3d.visualization.rendering.MaterialRecord()
        self.mtl.shader = "defaultUnlit"

    def capture(self, geoms, camera_position, camera_up):
        self.render.scene.scene.set_sun_light(
            geoms[0].get_center() + np.asarray(camera_position), [1.0, 1.0, 1.0], 75000
        )
        for i, item in enumerate(geoms):
            self.render.scene.add_geometry(f"{i}", item, self.mtl)

        camera = update_camera_position(
            self.camera,
            geoms[0].get_center() + np.asarray(camera_position),
            camera_target=geoms[0].get_center(),
            up=np.asarray(camera_up),
        )
        self.render.setup_camera(camera.intrinsic, camera.extrinsic)
        img = self.render.render_to_image()
        self.render.scene.clear_geometry()

        return img
