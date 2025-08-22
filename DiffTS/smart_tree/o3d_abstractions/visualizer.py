from dataclasses import asdict, dataclass
from typing import List, Sequence, Union

import open3d as o3d



@dataclass
class ViewerItem:
    name: str
    geometry: o3d.geometry.Geometry
    is_visible: bool = True


def o3d_viewer(
    items: Union[Sequence[ViewerItem], List[o3d.geometry.Geometry]], line_width=1
):
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = line_width

    if isinstance(items[0], o3d.geometry.Geometry):
        items = [ViewerItem(f"{i}", item) for i, item in enumerate(items)]

    def material(item):
        return line_mat if isinstance(item.geometry, o3d.geometry.LineSet) else mat

    geometries = [dict(**asdict(item), material=material(item)) for item in items]

    o3d.visualization.draw(geometries, line_width=line_width)
