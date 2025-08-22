import sys
import os
import open3d as o3d
from PyQt5 import QtWidgets, QtCore
import click

# Define a dictionary for data types
data_types = {
    'cond.ply': {'loader': o3d.io.read_point_cloud, 'color': [0, 0, 1], 'label': 'Show cond'},
    'gt.ply': {'loader': o3d.io.read_point_cloud, 'color': [0, 1, 0], 'label': 'Show gt'},
    'pred.ply': {'loader': o3d.io.read_point_cloud, 'color': [1, 0, 0], 'label': 'Show pred'},
    'gt_tube.ply': {'loader': o3d.io.read_triangle_mesh, 'color': [0, 1, 0], 'label': 'Show gt tubes'},
    'pred_open_edges.ply': {'loader': o3d.io.read_line_set, 'color': [1, 0, 1], 'label': 'Show pred open edges'},
    'pred_nn_edges.ply': {'loader': o3d.io.read_line_set, 'color': [1, 0, 0], 'label': 'Show pred nn edges'}
}

@click.command()
@click.option('--pcd_folder', '-p', type=str, help='Path to the folder containing PCD files', required=True)
def visualize_val(pcd_folder):
    file_list = sorted(os.listdir(pcd_folder))

    point_clouds = {key: [(f, data_types[key]['loader'](os.path.join(pcd_folder, f)).paint_uniform_color(data_types[key]['color'])) for f in file_list if f.endswith(key)] for key in data_types}

    app = QtWidgets.QApplication(sys.argv)
    viewer = PointCloudVisualizer(point_clouds)
    viewer.show()
    sys.exit(app.exec_())

class PointCloudVisualizer(QtWidgets.QWidget):
    def __init__(self, point_clouds):
        super().__init__()
        self.point_clouds = point_clouds
        self.current_index = 0
        self.init_ui()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Open3D', width=800, height=600, visible=True)
        self.update_point_cloud(0, reset_bounding_box=True)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_vis)
        self.timer.start(100)

    def init_ui(self):
        self.setWindowTitle('Point Cloud Viewer')
        self.setGeometry(100, 100, 800, 200)

        self.vbox = QtWidgets.QVBoxLayout()

        # Slider for selecting point clouds
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(next(iter(self.point_clouds.values()))) - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.update_point_cloud)

        self.vbox.addWidget(self.slider)

        # Label to show the current epoch
        self.epoch_label = QtWidgets.QLabel(f"Epoch: {self.current_index}")
        self.vbox.addWidget(self.epoch_label)

        # Checkboxes for enabling/disabling point clouds
        self.checkboxes = {}
        for key in data_types:
            checkbox = QtWidgets.QCheckBox(data_types[key]['label'])
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_point_cloud)
            self.checkboxes[key] = checkbox
            self.vbox.addWidget(checkbox)

        self.setLayout(self.vbox)

    def update_vis(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_point_cloud(self, index, reset_bounding_box=False):
        self.current_index = self.slider.value()
        self.epoch_label.setText(f"Epoch: {next(iter(self.point_clouds.values()))[self.current_index][0].split('_')[2]}")
        
        if hasattr(self, 'vis') and self.vis is not None:
            self.vis.clear_geometries()
            for key in data_types:
                if self.checkboxes[key].isChecked() and len(self.point_clouds[key]) > self.current_index:
                    self.vis.add_geometry(self.point_clouds[key][self.current_index][1], reset_bounding_box=reset_bounding_box)
            self.vis.poll_events()
            self.vis.update_renderer()

    def closeEvent(self, event):
        if hasattr(self, 'vis') and self.vis is not None:
            self.vis.close()
        event.accept()

if __name__ == "__main__":
    visualize_val()
