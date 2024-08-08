import numpy as np
from pytorch3d.ops import sample_points_from_meshes
import matplotlib.pyplot as plt
import matplotlib as mpl
import open3d as o3d
from mayavi import mlab
from pytorch3d.structures import Pointclouds
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_mesh_matplotlib(mesh):
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(190, 30)
    plt.show()


def visualize_mesh_open3d(mesh):
    vertices_cpu = mesh.verts_packed().detach().cpu().numpy()
    faces_cpu = mesh.faces_packed().detach().cpu().numpy()
    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_cpu)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces_cpu)
    colors = np.ones((vertices_cpu.shape[0], 3))
    colors[:, :] = [1, 0, 0]
    open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([open3d_mesh])


def visualize_height_map_mayavi(height_map, cuboid_points):
    mlab.figure(size=(800, 600), bgcolor=(0.16, 0.28, 0.46))
    mlab.surf(height_map, colormap='terrain', warp_scale='auto')
    mlab.colorbar(title='Height', orientation='vertical')
    mlab.axes(xlabel='X', ylabel='Y', zlabel='Height')
    mlab.view(azimuth=45, elevation=45, distance=600)
    mlab.points3d(cuboid_points[:, 0], cuboid_points[:, 1], cuboid_points[:, 2], scale_factor=0.3, color=(1, 0, 0))
    mlab.show()


def visualize_mesh_open3d_with_points(mesh, robot_point_cloud):
    vertices_cpu = mesh.verts_packed().detach().cpu().numpy()
    faces_cpu = mesh.faces_packed().detach().cpu().numpy()

    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_cpu)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces_cpu)
    colors = np.ones((vertices_cpu.shape[0], 3))
    colors[:, :] = [0, 1, 0]
    open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


    robot_point_cloud = robot_point_cloud.points_packed().cpu().numpy()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(robot_point_cloud)
    point_cloud.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([open3d_mesh, point_cloud])

