# from plyfile import PlyData

ply_filename = "/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/monogs-no-o3d/MonoGS/results/datasets_gtri/2024-06-06-12-14-28/point_cloud/final/point_cloud_copy.ply"
# f = open(ply_filename)
# data = PlyData(text=False, byte_order='<')
# data.read(f)
# print(data)

# f.close()

import io as _io
import numpy as np
from plyfile import PlyData  # Make sure plyfile is installed

def read_ply_file(filename, encoding='ISO-8859-1', error_handling='ignore'):
    """
    Read a PLY file with specified encoding and error handling.
    
    Parameters:
    - filename (str): Path to the PLY file.
    - encoding (str): The encoding to use for reading the file.
    - error_handling (str): The error handling scheme ('ignore', 'replace', 'strict', etc.).
    
    Returns:
    - PlyData: The parsed PLY data.
    """
    with open(filename, 'r', encoding=encoding, errors=error_handling) as file:
        data = file.read()
    return PlyData.read(_io.StringIO(data))

# Example usage:
# ply_data = read_ply_file(ply_filename, encoding='ISO-8859-1')

from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def load_ply(path):
    plydata = PlyData.read(path)

    def fetchPly_nocolor(path):
        plydata = PlyData.read(path)
        vertices = plydata["vertex"]
        positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
        colors = np.ones_like(positions)
        return BasicPointCloud(points=positions, colors=colors, normals=normals)

    ply_input = fetchPly_nocolor(path)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    print(opacities)

ply_data = load_ply(ply_filename)