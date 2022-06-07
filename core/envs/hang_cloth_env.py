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

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from gym.spaces import Box
from jax import jit, vmap, random

from core.engine.cloth_simulator_gripper import robot_step, create_vars, get_x_grid, get_indices
from core.engine.mpm_simulator_md import process_state
from core.engine.render import MeshRenderer

N = 64
size = int(N / 5)
pole_pos = np.array([0., 0.3, 0.15])
pole_radius = 0.01


class HangCloth:

    def __init__(self, robot_step_grad_fun, max_steps, init_state, batch_size, visualize=False):
        self.state = None
        self.seed_num = 0
        self.key = random.PRNGKey(self.seed_num)
        self.init_state = init_state
        self.batch_size = batch_size
        self.step_jax = robot_step_grad_fun
        self.max_steps = max_steps
        self.cur_step = 0
        self.action_size = 8
        self.observation_size = 288 * 6 + 8
        self.cloth_state_shape = (288, 6)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(288 * 6 + 8,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.visualize = visualize
        self.spec = None

        cloth_mask = create_cloth_mask()
        self.idx_i, self.idx_j = jnp.nonzero(cloth_mask)

        if visualize:
            import open3d as o3d
            renderer = MeshRenderer()
            pole = o3d.geometry.TriangleMesh.create_cylinder(radius=pole_radius, height=0.35)
            R = pole.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
            pole.rotate(R, center=(0, 0, 0))
            pole.translate([0.5, pole_pos[1], pole_pos[2]])
            pole.compute_vertex_normals()
            pole.paint_uniform_color([0.6, 0.6, 0.6])
            renderer.vis.add_geometry(pole)

            # add grippers
            ps0_np = np.array(init_state[2])
            gripper0 = o3d.geometry.TriangleMesh.create_sphere(radius=ps0_np[3])
            gripper0.translate(ps0_np[:3], relative=False)
            renderer.vis.add_geometry(gripper0)

            ps1_np = np.array(init_state[3])
            gripper1 = o3d.geometry.TriangleMesh.create_sphere(radius=ps1_np[3])
            gripper1.translate(ps1_np[:3], relative=False)
            renderer.vis.add_geometry(gripper1)

            self.renderer = renderer
            self.gripper0 = gripper0
            self.gripper1 = gripper1

    def seed(self, seed):
        self.seed_num = seed
        self.key = random.PRNGKey(self.seed_num)
        self.init_state = self.init_state[:-1] + (self.key,)
        self.reset()

    def step(self, action):
        self.state = self.step_jax(action, *self.state)

        obs = jnp.concatenate([self.state[0], self.state[1]], axis=-1)
        obs = jnp.concatenate([obs.flatten(), self.state[2], self.state[3]], axis=-1)
        # TODO change observations
        reward, done, info = 0, self.cur_step > self.max_steps, {}

        if self.cur_step >= self.max_steps - 1:
            done = True
            x = self.state[0]
            if x[:, 1].max() >= pole_pos[1] and x[:, 2].min() <= pole_pos[2] and x[:, 2].max() >= pole_pos[2]:
                reward = 1.

        self.cur_step += 1

        return np.array(obs), reward, done, info

    def reset(self):
        cloth_mask = create_cloth_mask()
        self.key, _ = jax.random.split(self.key)
        state = create_vars(N, collision_func, cloth_mask, self.key)

        actions = jnp.zeros((60, 8))
        for action in actions:
            state = self.step_jax(action, *state)

        x, v, ps0, ps1, key = state
        ps0 = [4.0625378e-01, -4.9900409e-04, 5.1644766e-01, 0.01]
        ps1 = [5.312532e-01, -4.990041e-04, 5.164433e-01, 0.01]
        ps0 = jnp.array(ps0)
        ps1 = jnp.array(ps1)
        self.state = (x, v, ps0, ps1, key)

        obs = jnp.concatenate([self.state[0], self.state[1]], axis=-1)
        obs = jnp.concatenate([obs.flatten(), self.state[2], self.state[3]], axis=-1)

        return np.array(obs)

    @staticmethod
    def reset_jax(key_envs, step_jax, batch_size):
        cloth_mask = create_cloth_mask()
        key_envs, _ = jax.random.split(key_envs)
        state = create_vars(N, collision_func, cloth_mask, key_envs)

        state = process_state(state, v_size=batch_size, p_size=0)
        actions = jnp.zeros((60, batch_size, 8)) if batch_size > 0 else jnp.zeros((60, 8))
        for action in actions:
            state = step_jax(action, *state)

        x, v, ps0, ps1, key = state
        ps0 = [4.0625378e-01, -4.9900409e-04, 5.1644766e-01, 0.01]
        ps1 = [5.312532e-01, -4.990041e-04, 5.164433e-01, 0.01]
        ps0 = jnp.array(ps0) if batch_size == 0 else jnp.array([ps0] * batch_size)
        ps1 = jnp.array(ps1) if batch_size == 0 else jnp.array([ps1] * batch_size)
        state = (x, v, ps0, ps1, key)

        return state

    def render(self):
        x, v, ps0, ps1, _ = self.state
        indices = get_indices()
        x_grid_ = get_x_grid(x)
        x_grid_ = x_grid_.reshape((-1, 3))

        self.gripper0.translate(np.array(ps0)[:3], relative=False)
        self.renderer.vis.update_geometry(self.gripper0)
        self.gripper1.translate(np.array(ps1)[:3], relative=False)
        self.renderer.vis.update_geometry(self.gripper1)

        self.renderer.render(0, vertices=x_grid_, indices=indices)


def collision_func(x, v, idx_i, idx_j):
    # collision with pole
    x = jnp.zeros((N, N, 3)).at[idx_i, idx_j].set(x)
    v = jnp.zeros((N, N, 3)).at[idx_i, idx_j].set(v)

    # find points on the surface
    x -= jnp.array(pole_pos).reshape(1, 1, 3)
    mask = jnp.linalg.norm(x[..., 1:], axis=-1) <= pole_radius
    mask = mask[..., None].repeat(3, -1)

    surface_norm = x.at[..., 0].set(0) * -1  # point into the pole

    # calc surface norm at each point
    norm_ = jnp.sqrt((surface_norm ** 2).sum(-1))[..., None]
    dot_prod = jnp.einsum('ijk,ijk->ij', v, surface_norm).clip(0, jnp.inf)[..., None]
    proj_of_v_on_surface = (dot_prod / norm_ ** 2) * surface_norm
    v_ = v - proj_of_v_on_surface  # prevent from going into the pole

    v_ *= 0.95  # simulate friction
    v = jnp.where(mask, v_, v)

    v = v[idx_i, idx_j]
    return v


def create_cloth_mask():
    cloth_mask = jnp.zeros((N, N))
    cloth_mask = cloth_mask.at[size * 2:size * 3, size * 2:size * 4].set(1)

    return cloth_mask


def make_env(batch_size=0, episode_length=80, visualize=False, seed=0):
    cloth_mask = create_cloth_mask()
    key = random.PRNGKey(seed)
    state = create_vars(N, collision_func, cloth_mask, key)

    actions = jnp.zeros((100, 8))

    # compile sim according to conf
    state = process_state(state, v_size=batch_size, p_size=0)
    robot_step_grad_fun = robot_step
    if batch_size > 0:
        actions = jnp.array(actions)[:, None, ...].repeat(batch_size, 1)
        robot_step_grad_fun = vmap(robot_step_grad_fun)
    print("compiling simulation")
    robot_step_grad_fun = jit(robot_step_grad_fun)
    robot_step_grad_fun(actions[0], *state)  # to warm up

    env = HangCloth(robot_step_grad_fun, max_steps=episode_length,
                    init_state=state, batch_size=batch_size, visualize=visualize)

    return env


if __name__ == "__main__":
    print(jax.devices())
    env = make_env(batch_size=0, visualize=True)
    env.reset()
