import numpy as np
from pytorch3d.ops import sample_points_from_meshes
import matplotlib.pyplot as plt
import matplotlib as mpl
import open3d as o3d
from mayavi import mlab
import pyvista as pv
from pytorch3d.structures import Pointclouds
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80


def visualize_two_py_torch3d_meshes(mesh1, mesh2):
    vertices1_cpu = mesh1.verts_packed().detach().cpu().numpy()
    faces1_cpu = mesh1.faces_packed().detach().cpu().numpy()
    faces1_pv = np.hstack([[3, *face] for face in faces1_cpu])
    mesh1_pv = pv.PolyData(vertices1_cpu, faces1_pv)
    mesh1_pv['colors'] = [255] * vertices1_cpu.shape[0]
    vertices2_cpu = mesh2.verts_packed().detach().cpu().numpy()
    faces2_cpu = mesh2.faces_packed().detach().cpu().numpy()
    faces2_pv = np.hstack([[3, *face] for face in faces2_cpu])
    mesh2_pv = pv.PolyData(vertices2_cpu, faces2_pv)
    mesh2_pv['colors'] = [200] * vertices2_cpu.shape[0]
    x_axis = pv.Line((25, 25, 0), (40, 25, 0))
    x_axis = x_axis.tube(radius=0.05)
    y_axis = pv.Line((25, 25, 0), (25, 40, 0))
    y_axis = y_axis.tube(radius=0.05)
    z_axis = pv.Line((25, 25, 0), (25, 25, 15))
    z_axis = z_axis.tube(radius=0.05)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh1_pv, show_edges=True)
    plotter.add_mesh(mesh2_pv, show_edges=True)
    plotter.add_mesh(x_axis, color='red')
    plotter.add_mesh(y_axis, color='green')
    plotter.add_mesh(z_axis, color='blue')
    labels = dict(zlabel='z', xlabel='x', ylabel='y')
    plotter.show_grid(**labels)
    plotter.add_axes(**labels)
    plotter.show()
