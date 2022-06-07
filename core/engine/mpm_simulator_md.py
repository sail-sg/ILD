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

import jax.numpy as jnp


def process_state(state, v_size=0, p_size=0):
    """
    :param state: tuple of vars
    :return: convert into device arrays with additional batch dim
    """

    state_new = ()
    for i in range(len(state)):

        var_tmp = jnp.array(state[i])

        if v_size > 0:
            var_tmp = var_tmp[None, ...].repeat(v_size, 0)

        if p_size > 0:
            var_tmp = var_tmp[None, ...].repeat(p_size, 0)

        state_new += (var_tmp,)

    return state_new
