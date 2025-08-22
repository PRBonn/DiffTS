
import numpy as np

import wandb


def render_cloud(
    renderer,
    labelled_cloud,
    camera_position=[1, 0, 0],
    camera_up=[0, 1, 0],
):
    segmented_img = renderer.capture(
        [labelled_cloud.to_o3d_seg_cld()],
        camera_position,
        camera_up,
    )

    cld_img = renderer.capture(
        [labelled_cloud.to_o3d_cld()],
        camera_position,
        camera_up,
    )

    projected_img = renderer.capture(
        [labelled_cloud.to_o3d_medial_vectors()],
        camera_position,
        camera_up,
    )

    return [
        np.asarray(cld_img),
        np.asarray(segmented_img),
        np.asarray(projected_img),
    ]


def log_images(wandb_run, name, images, epoch):
    wandb_run.log({f"{name}": [wandb.Image(img) for img in images]})
