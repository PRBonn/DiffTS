from smart_tree.data_types.tree import (DisjointTreeSkeleton, TreeSkeleton,
                                        connect_skeletons)
from smart_tree.o3d_abstractions.visualizer import o3d_viewer

if __name__ == "__main__":
    disjoint_skeleton = DisjointTreeSkeleton.from_pickle(
        "/local/smart-tree/test_data/disjoint_skeleton.pkl"
    )

    # disjoint_skeleton.prune(min_radius=0.01, min_length=0.08).smooth(kernel_size=11)
    # Sort skeletons by total length
    skeletons_sorted = sorted(
        disjoint_skeleton.skeletons,
        key=lambda x: x.length,
        reverse=True,
    )

    # skeletons_sorted[0].view()

    skel = connect_skeletons(skeletons_sorted[0], 0, 0, skeletons_sorted[1], 0, 0)

    skel.view()

    quit()

    final_skeleton = TreeSkeleton(0, skeletons_sorted[0].branches)

    for skeleton in skeletons_sorted[1:]:
        branch = skeleton.branches[skeleton.key_branch_with_biggest_radius]

        # get the point that has the biggest radius ....
        # get the closest point on the skeleton to that point ...
        # connect the two points ...

        print(skeleton.length)

        o3d_viewer(
            [final_skeleton.to_o3d_tube(), branch.to_o3d_tube(), skeleton.to_o3d_tube()]
        )
