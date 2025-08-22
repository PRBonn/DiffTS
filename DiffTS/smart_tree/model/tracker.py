import numpy as np

import wandb


class Tracker:
    def __init__(self):
        self.running_epoch_radius_loss = []
        self.running_epoch_direction_loss = []
        self.running_epoch_class_loss = []

    def update(self, loss_dict: dict):
        self.running_epoch_radius_loss.append(loss_dict["radius"].item())
        self.running_epoch_direction_loss.append(loss_dict["direction"].item())
        self.running_epoch_class_loss.append(loss_dict["class_l"].item())

    @property
    def radius_loss(self):
        return np.mean(self.running_epoch_radius_loss)

    @property
    def direction_loss(self):
        return np.mean(self.running_epoch_direction_loss)

    @property
    def class_loss(self):
        return np.mean(self.running_epoch_class_loss)

    @property
    def total_loss(self):
        return self.radius_loss + self.direction_loss + self.class_loss

    def log(self, name, epoch):
        wandb.log(
            {
                f"{name} Total Loss": self.total_loss,
                f"{name} Radius Loss": self.radius_loss,
                f"{name} Direction Loss": self.direction_loss,
                f"{name} Class Loss": self.class_loss,
            },
            epoch,
        )
