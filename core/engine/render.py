# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import pathlib

import numpy as np


my_path = pathlib.Path(__file__).parent.resolve()

def draw_xyz(x, y, z, l):
    import open3d as o3d
    points = [
        [x, y, z],
        [x + l, y, z],
        [x, y + l, z],
        [x, y, z + l], ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
    ]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_container(size, pos, rot):
    import open3d as o3d
    # size = np.array(size) - 0.02  # only draw inner wall
    points = [
        [size[0], -size[1], size[2]],
        [size[0], -size[1], -size[2]],
        [-size[0], -size[1], -size[2]],
        [-size[0], -size[1], size[2]],

        [size[0], size[1], size[2]],
        [size[0], size[1], -size[2]],
        [-size[0], size[1], -size[2]],
        [-size[0], size[1], size[2]],
    ]

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],

        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0]] * 8
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    rot_mat = line_set.get_rotation_matrix_from_quaternion(rot)
    line_set = line_set.rotate(rot_mat)
    line_set = line_set.translate(pos)

    return line_set


def visualize_pc(xyz):
    import open3d as o3d
    xyz = xyz[xyz[:, 2] > -5]
    xyz = xyz[xyz[:, 1] > -5]
    xyz = xyz[xyz[:, 0] > -5]
    xyz = xyz[xyz[:, 2] < 5]
    xyz = xyz[xyz[:, 1] < 5]
    xyz = xyz[xyz[:, 0] < 5]

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(xyz)
    origin_xyz = draw_xyz(0, 0, 0, 1)
    o3d.visualization.draw_geometries([pcd_o3d, origin_xyz])


class BasicRenderer:

    def __init__(self, box_sizes, colors):
        import open3d as o3d
        self.box_sizes = np.array(box_sizes)
        self.n_primitives = len(box_sizes)
        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.pcd_o3d)

        origin_xyz = draw_xyz(0, 0, 0, 1)
        self.vis.add_geometry(origin_xyz)

        self.containers = []
        self.containers_ = []

        for i in range(self.n_primitives):
            box_size = box_sizes[i]
            # mesh_box = o3d.geometry.TriangleMesh.create_box(width=box_size[0] * 2, height=box_size[1] * 2,
            #                                                 depth=box_size[2] * 2)
            # mesh_box.compute_vertex_normals()
            # mesh_box.paint_uniform_color([0.1, 0.1, 0.1])
            container = draw_container(box_size, [0, 0, 0], [1, 0, 0, 0])
            self.containers.append(container)

            container_ = copy.deepcopy(container)
            self.containers_.append(container_)
            self.vis.add_geometry(container_)

    def render(self, i, state):
        for i in range(self.n_primitives):
            obj_pos = np.array(state[12][i].reshape((-1, 3))[0])
            obj_rot = np.array(state[13][i].reshape((-1, 4))[0])
            obj_rot = self.containers[i].get_rotation_matrix_from_quaternion(obj_rot)
            # effector_pos -= self.box_sizes[i]
            self.vis.remove_geometry(self.containers_[i])
            self.containers_[i] = copy.deepcopy(self.containers[i]).rotate(obj_rot).translate(obj_pos)
            self.vis.add_geometry(self.containers_[i])

            # obj_rot = self.containers[i].get_rotation_matrix_from_quaternion(obj_rot)
            # self.containers[i].rotate(default_rot).translate([0, 0, 0])
            # self.containers[i].rotate(obj_rot).translate(obj_pos)
            # self.vis.update_geometry(self.containers[i])

        obj_pos = np.array(state[12].reshape((-1, 3))[0])
        particle_pos = np.array(state[0].reshape((-1, 3))[:state[0].shape[-2]])
        print(i, "position", obj_pos, particle_pos[0])
        self.pcd_o3d.points = self.o3d.utility.Vector3dVector(particle_pos)
        self.vis.update_geometry(self.pcd_o3d)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.poll_events()


class MeshRenderer:

    def __init__(self):
        import open3d as o3d
        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        render_op = self.vis.get_render_option()
        render_op.mesh_show_wireframe = True
        render_op.mesh_show_back_face = True

        self.mesh = o3d.geometry.TriangleMesh()
        self.vis.add_geometry(self.mesh)

        # origin_xyz = draw_xyz(0, 0, 0, 1)
        # self.vis.add_geometry(origin_xyz)



    def render(self, i, vertices, indices):
        import open3d as o3d
        np_vertices = np.array(vertices)
        np_triangles = np.array(indices).astype(np.int32).reshape((-1, 3))
        self.mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(np_triangles)

        self.mesh.compute_vertex_normals()
        self.vis.update_geometry(self.mesh)
        self.vis.poll_events()
        self.vis.update_renderer()
