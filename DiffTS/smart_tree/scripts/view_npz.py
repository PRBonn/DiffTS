import argparse
from pathlib import Path
from typing import List, Tuple


from smart_tree.data_types.cloud import Cloud
from smart_tree.data_types.tree import TreeSkeleton
from smart_tree.o3d_abstractions.visualizer import ViewerItem, o3d_viewer
from smart_tree.util.file import load_data_npz


def view_synthetic_data(data: List[Tuple[Cloud, TreeSkeleton]], line_width=1):
    geometries = []
    for i, item in enumerate(data):
        (cloud, skeleton), path = item

        tree_name = path.stem
        visible = i == 0

        geometries = [
            ViewerItem(
                f"{tree_name}_cloud",
                cloud.to_o3d_cld(),
                is_visible=visible,
            ),
            ViewerItem(
                f"{tree_name}_labelled_cloud",
                cloud.to_o3d_seg_cld(),
                is_visible=visible,
            ),
            ViewerItem(
                f"{tree_name}_medial_vectors",
                cloud.to_o3d_medial_vectors(),
                is_visible=visible,
            ),
            # ViewerItem(
            #     f"{tree_name}_skeleton",
            #     skeleton.to_o3d_lineset(),
            #     is_visible=visible,
            # ),
            # ViewerItem(
            #     f"{tree_name}_skeleton_mesh",
            #     skeleton.to_o3d_tubes(),
            #     is_visible=visible,
            # ),
        ]

    o3d_viewer(geometries, line_width=line_width)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualizer Arguments")

    parser.add_argument(
        "file_path",
        help="File Path of tree.npz",
        default=None,
        type=Path,
    )

    parser.add_argument(
        "-lw",
        "--line_width",
        help="Width of visualizer lines",
        default=1,
        type=int,
    )
    return parser.parse_args()


def paths_from_args(args, glob="*.npz"):
    if not args.file_path.exists():
        raise ValueError(f"File {args.file_path} does not exist")

    if args.file_path.is_file():
        print(f"Loading data from file: {args.file_path}")
        return [args.file_path]

    if args.file_path.is_dir():
        print(f"Loading data from directory: {args.file_path}")
        files = args.file_path.glob(glob)
        if files == []:
            raise ValueError(f"No npz files found in {args.file_path}")
        return files


def main():
    args = parse_args()

    data = [(load_data_npz(filename), filename) for filename in paths_from_args(args)]
    view_synthetic_data(data, args.line_width)


if __name__ == "__main__":
    main()
