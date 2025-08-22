import laspy
import numpy as np
import open3d as o3d


def las_to_ply(input_las_file, output_ply_file):
    with laspy.open(input_las_file) as fh:
        las = fh.read()
        xyz = np.column_stack((las.x, las.y, las.z))

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    o3d.io.write_point_cloud(output_ply_file, cloud)


if __name__ == "__main__":
    input_las_file = "/csse/users/hdo27/Desktop/ChCh_Hovermap_tree.laz"
    output_ply_file = "output.ply"

    las_to_ply(input_las_file, output_ply_file)
