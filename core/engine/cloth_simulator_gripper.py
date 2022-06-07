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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import custom_vjp, random
from jax._src.lax.control_flow import fori_loop, scan

N = 190
cell_size = 1.0 / N
gravity = 0.5
stiffness = 1600
damping = 2
dt = 2e-3
max_v = 2.

ball_radius = 0.2
num_triangles = (N - 1) * (N - 1) * 2

links = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]]
links = jnp.array(links)

a, b = jnp.indices((N, N))
grid_idx = jnp.concatenate((a[..., None], b[..., None]), axis=2)
indices, vertices, cloth_mask = None, None, None
key_global = jax.random.PRNGKey(1)


@partial(custom_vjp)
def norm_grad(x):
    return x


def norm_grad_fwd(x):
    return x, ()


def norm_grad_bwd(x, g):
    g /= jnp.linalg.norm(g)
    g = jnp.nan_to_num(g)
    g /= cloth_mask.sum()

    return g,


norm_grad.defvjp(norm_grad_fwd, norm_grad_bwd)


def default_collision_func(x, v, idx_i, idx_j):
    return x


def primitive_collision_func(x, v, action, ps):
    # collision with primitive ball
    pos, radius = ps[:3], ps[3]
    d_v = action[:3].reshape(1, 3)
    suction = action[-1]

    # find points on the surface
    x_ = x - jnp.array(pos).reshape(1, 3)
    dist = jnp.linalg.norm(x_, axis=-1)
    mask = dist <= radius
    mask = mask[..., None].repeat(3, -1)
    v_ = jnp.where(mask, 0, v)
    x_ = jnp.where(mask, x + d_v * (1 - suction), x)

    # weight = jnp.exp(-1 * (dist*20 - 1))[..., None]
    # v = v - weight * suction * v
    # x = x + d_v * weight

    v_mask = jnp.abs(v).max() > max_v
    v = jnp.where(v_mask, v, v_)
    x = jnp.where(v_mask, x, x_)

    x = norm_grad(x)
    v = norm_grad(v)

    return x, v


def create_vars(N_, collision_func_, cloth_mask_, key_):
    global N, num_triangles, grid_idx, cell_size, collision_func, \
        indices, vertices, cloth_mask, idx_i, idx_j, x_grid

    # set global vars
    N = N_
    num_triangles = (N - 1) * (N - 1) * 2
    cell_size = 1.0 / N
    collision_func = collision_func_

    # cloth mask
    indices = jnp.zeros((num_triangles * 3,))
    vertices = jnp.zeros((N * N, 3))
    cloth_mask = cloth_mask_
    idx_i, idx_j = jnp.nonzero(cloth_mask)
    grid_idx = jnp.concatenate([idx_i[:, None], idx_j[:, None]], axis=-1)

    # create x, v
    x = np.zeros((N, N, 3))
    for i, j in np.ndindex((N, N)):
        x[i, j] = np.array([
            i * cell_size, j * cell_size / np.sqrt(2),
            (N - j) * cell_size / np.sqrt(2) + 0.1
        ])
    x_grid = jnp.array(x)
    v = jnp.zeros((N, N, 3))
    ps0 = jnp.array([0., 0.1, 0.58, 0.01])
    ps1 = jnp.array([0., 0.1, 0.58, 0.01])

    set_indices()

    # mask x and v
    x = x_grid[idx_i, idx_j]
    v = v[idx_i, idx_j]

    return x, v, ps0, ps1, key_


def set_indices():
    global indices, cloth_mask
    indices = np.array(indices)
    cloth_mask = np.array(cloth_mask)
    for i, j in np.ndindex((N, N)):

        if i < N - 1 and j < N - 1:
            flag = 1
            flag *= cloth_mask[i - 1, j - 1] * cloth_mask[i - 1, j] * cloth_mask[i - 1, j + 1]
            flag *= cloth_mask[i, j - 1] * cloth_mask[i, j] * cloth_mask[i, j + 1]
            flag *= cloth_mask[i + 1, j - 1] * cloth_mask[i + 1, j] * cloth_mask[i + 1, j + 1]

            square_id = (i * (N - 1)) + j
            # 1st triangle of the square
            indices[square_id * 6 + 0] = i * N + j
            indices[square_id * 6 + 1] = (i + 1) * N + j
            indices[square_id * 6 + 2] = i * N + (j + 1)
            # 2nd triangle of the square
            indices[square_id * 6 + 3] = (i + 1) * N + j + 1
            indices[square_id * 6 + 4] = i * N + (j + 1)
            indices[square_id * 6 + 5] = (i + 1) * N + j

            indices[square_id * 6 + 0] *= flag
            indices[square_id * 6 + 1] *= flag
            indices[square_id * 6 + 2] *= flag
            # 2nd triangle of the square
            indices[square_id * 6 + 3] *= flag
            indices[square_id * 6 + 4] *= flag
            indices[square_id * 6 + 5] *= flag

    cloth_mask = jnp.array(cloth_mask)
    indices = jnp.array(indices)
    indices = indices.reshape((-1, 3))
    indices = indices[indices.sum(1) != 0]


def robot_step(action, x, v, ps0, ps1, key):
    def step_(i, state):
        action, _, _, _, _, _ = state
        state_ = step(*state)
        return (action,) + state_

    # normalize speed, 50 sub steps, 20 is a scale factor
    action = action.at[:3].set(action[:3].clip(-1, 1) / 50. / 20.)
    action = action.at[4:7].set(action[4:7].clip(-1, 1) / 50. / 20.)

    # add uncertainty
    key, _ = random.split(key)
    action += random.uniform(key, action.shape) * 0.0001 - 0.00005  # randomness
    action += random.uniform(key_global, action.shape) * 0.0004 - 0.0002  # fixed bias, as key_global won't change

    state = (action, x, v, ps0, ps1, key)
    state = fori_loop(0, 50, step_, state)

    return state[1:]


def step(action, x, v, ps0, ps1, key):
    v -= jnp.array([0, gravity * dt, 0])
    action = action.clip(-1, 1)

    action = action.at[3].set(0)
    action = action.at[7].set(0)

    # mask out invalid area
    # v *= cloth_mask.reshape((N, N, 1))
    j_ = grid_idx.reshape((-1, 1, 2)).repeat(len(links), -2)
    j_ = j_ + links[None, ...]
    j_ = jnp.clip(j_, 0, N - 1)

    i_ = grid_idx.reshape((-1, 1, 2)).repeat(len(links), -2)
    original_length = cell_size * jnp.linalg.norm(j_ - i_, axis=-1)[..., None]
    ori_len_is_not_0 = (original_length != 0).astype(jnp.int32)
    original_length = jnp.clip(original_length, 1e-12, jnp.inf)

    j_x, j_y = j_.reshape((-1, 2))[:, 0], j_.reshape((-1, 2))[:, 1]
    i_x, i_y = i_.reshape((-1, 2))[:, 0], i_.reshape((-1, 2))[:, 1]

    x_grid = jnp.zeros((N, N, 3)).at[idx_i, idx_j].set(x)
    relative_pos = x_grid[j_x, j_y] - x_grid[i_x, i_y]
    # current_length = jnp.linalg.norm(relative_pos, axis=-1)
    current_length = jnp.clip((relative_pos ** 2).sum(-1), 1e-12, jnp.inf) ** 0.5
    current_length = current_length.reshape((-1, len(links), 1))

    force = stiffness * relative_pos.reshape((-1, 8, 3)) / current_length * (
            current_length - original_length) / original_length

    force *= ori_len_is_not_0

    # mask out force from invalid area
    force *= cloth_mask[j_x, j_y].reshape((-1, 8, 1))

    force = force.sum(1)
    v += force * dt
    v *= jnp.exp(-damping * dt)

    # collision
    v = collision_func(x, v, idx_i, idx_j)
    x, v = primitive_collision_func(x, v, action[:4], ps0)
    x, v = primitive_collision_func(x, v, action[4:], ps1)

    v_mask = jnp.abs(v).max() > max_v
    ps0_ = ps0.at[:3].add(action[:3]).clip(0, 1)
    ps1_ = ps1.at[:3].add(action[4:7]).clip(0, 1)
    ps0 = jnp.where(v_mask, ps0, ps0_)
    ps1 = jnp.where(v_mask, ps1, ps1_)

    # collision with the ground
    x = x.clip(0, 1)
    v = v.clip(-max_v, max_v)

    x += dt * v

    x = norm_grad(x)
    v = norm_grad(v)
    ps0 = norm_grad(ps0)
    ps1 = norm_grad(ps1)

    return x, v, ps0, ps1, key


def get_indices():
    return indices


def get_x_grid(x):
    x_grid_ = x_grid.at[idx_i, idx_j].set(x)
    return x_grid_


collision_func = default_collision_func
