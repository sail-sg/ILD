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

import os

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
import sys
import time
from functools import partial
from typing import Any, Callable, Dict, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl import logging
from brax import envs
from brax.training import distribution, networks
from brax.training import normalization
from brax.training.types import PRNGKey
from brax.training.types import Params
from flax import linen
from jax import custom_vjp
sys.path.append('./../..')
from core.envs.hang_cloth_env import make_env as make_env_hang_cloth, pole_pos

logging.set_verbosity(logging.INFO)

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    key: PRNGKey
    optimizer_state: optax.OptState
    il_optimizer_state: optax.OptState
    policy_params: Params


def group_state(state):
    cloth_state = jnp.concatenate([state[0], state[1]], axis=-1)
    cloth_state = cloth_state.reshape(cloth_state.shape[:-2] + (-1,))

    gripper_state = jnp.concatenate([state[2], state[3]], axis=-1)
    combo_state = jnp.concatenate([cloth_state, gripper_state], axis=-1)

    return combo_state


def train(
        environment_fn: Callable[..., envs.Env],
        episode_length: int,
        action_repeat: int = 1,
        num_envs: int = 1,
        num_eval_envs: int = 128,
        max_gradient_norm: float = 1e9,
        max_devices_per_host: Optional[int] = None,
        learning_rate=1e-4,
        normalize_observations=False,
        seed=0,
        log_frequency=10,
        progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        truncation_length: Optional[int] = None,
):
    xt = time.time()

    # prepare expert demos
    args.logdir = f"logs/{args.env}/{args.env}_ep_len{args.ep_len}_num_envs{args.num_envs}_lr{args.lr}_trunc_len{args.trunc_len}" \
                  f"_max_it{args.max_it}_max_grad_norm{args.max_grad_norm}_re_dis{args.reverse_discount}_ef_{args.entropy_factor}" \
                  f"_df_{args.deviation_factor}_acf_{args.action_cf_factor}" \
                  f"/seed{args.seed}"
    demo_traj_raw = np.load(f"expert/{args.env}_traj_state.npy", allow_pickle=True)
    demo_traj_raw = jnp.array(demo_traj_raw)[:args.ep_len][:, None, ...].repeat(args.num_envs, 1)
    demo_gripper_traj = demo_traj_raw[..., -10:]
    demo_state_traj = demo_traj_raw[..., :-10]

    demo_traj_action = np.load(f"expert/{args.env}_traj_action.npy")
    demo_traj_action = jnp.array(demo_traj_action)[:args.ep_len][:, None, ...].repeat(args.num_envs, 1)

    reverse_discounts = jnp.array([args.reverse_discount ** i for i in range(args.ep_len, 0, -1)])[None, ...]
    reverse_discounts = reverse_discounts.repeat(args.num_envs, 0)

    # tensorboard
    file_writer = tf.summary.create_file_writer(args.logdir)
    file_writer.set_as_default()

    # distributed training setup
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info('Device count: %d, process count: %d (id %d), local device count: %d, '
                 'devices to be used count: %d', jax.device_count(), process_count,
                 process_id, local_device_count, local_devices_to_use)
    logging.info('Available devices %s', jax.devices())

    # seeds
    key = jax.random.PRNGKey(seed)
    key, key_models, key_env = jax.random.split(key, 3)
    key_debug = jax.random.PRNGKey(seed + 666)

    # envs
    core_env = environment_fn(batch_size=num_envs, episode_length=episode_length)
    step_fn = core_env.step_jax
    reset_fn = core_env.reset_jax
    first_state = reset_fn(key_env, step_fn, args.num_envs)

    eval_env = environment_fn(batch_size=num_eval_envs, episode_length=episode_length)
    eval_step_fn = eval_env.step_jax
    eval_first_state = eval_env.reset_jax(key_env, eval_step_fn, num_eval_envs)

    visualize_env = environment_fn(batch_size=0, episode_length=episode_length, visualize=False)
    visualize_env.step_jax = visualize_env.step_jax
    visualize_first_state = visualize_env.reset_jax(key_env, visualize_env.step_jax, batch_size=0)

    # initialize policy
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=core_env.action_size)
    policy_model = make_direct_optimization_model(parametric_action_distribution, core_env.observation_size)

    # init optimizer
    policy_params = policy_model.init(key_models)
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(policy_params)
    il_optimizer_state = optimizer.init(policy_params)

    """
    IL boostrap
    """

    def il_loss(params, key):

        logits = policy_model.apply(params, demo_traj_raw.reshape((-1, demo_traj_raw.shape[-1])))
        logits = logits.reshape(demo_traj_raw.shape[:2] + (-1,))
        rollout_actions = parametric_action_distribution.sample(logits, key)

        loss_val = (rollout_actions - demo_traj_action) ** 2
        loss_val = jnp.sqrt(loss_val.sum(-1)).mean()
        return loss_val, loss_val

    def il_minimize(training_state: TrainingState):

        grad, loss_val = il_loss_grad(training_state.policy_params, training_state.key)
        grad = clip_by_global_norm(grad)
        params_update, il_optimizer_state = optimizer.update(grad, training_state.il_optimizer_state)
        policy_params = optax.apply_updates(training_state.policy_params, params_update)

        metrics = {
            'grad_norm': optax.global_norm(grad),
            'params_norm': optax.global_norm(policy_params),
            'loss_val': loss_val
        }
        return TrainingState(
            key=key,
            optimizer_state=training_state.optimizer_state,
            il_optimizer_state=il_optimizer_state,
            policy_params=policy_params), metrics, loss_val

    """
    Evaluation functions
    """

    def visualize(params, state, key):
        if not visualize_env.visualize:
            return

        visualize_env.reset()
        for i in range(args.ep_len):
            key, key_sample = jax.random.split(key)
            combo_state = group_state(state)[None, ...]
            logits = policy_model.apply(params, combo_state)
            logits = logits.reshape((combo_state.shape[0], -1))
            actions = parametric_action_distribution.sample(logits, key_sample)
            actions = actions.squeeze()

            state = visualize_env.step_jax(actions, *state)
            visualize_env.state = state
            visualize_env.render()

    def do_one_step_eval(carry, unused_target_t):
        state, params, key = carry
        key, key_sample = jax.random.split(key)
        combo_state = group_state(state)

        print(jax.tree_map(lambda x: x.shape, params))
        logits = policy_model.apply(params, combo_state)
        logits = logits.reshape((combo_state.shape[0], -1))
        actions = parametric_action_distribution.sample(logits, key_sample)
        nstate = eval_step_fn(actions, *state)
        return (nstate, params, key), state

    @jax.jit
    def run_eval(params, state, key):
        (state, _, key), state_list = jax.lax.scan(
            do_one_step_eval, (state, params, key), (),
            length=episode_length // action_repeat)
        demo_traj_raw_ = demo_traj_raw[:, 0, :][:, None, :].repeat(num_eval_envs, 1)

        combo_state = group_state(state_list)
        combo_state = jnp.nan_to_num(combo_state)

        loss_val = (combo_state - demo_traj_raw_) ** 2
        loss_val = loss_val.sum(-1).mean()

        return state, key, state_list, loss_val

    def eval_policy(it, key_debug):
        state, key, state_list, loss_val = run_eval(training_state.policy_params, eval_first_state, key_debug)

        x = jnp.nan_to_num(state[0])
        reward = (x[:, :, 1].max(1) >= pole_pos[1]).astype(jnp.float32)
        reward *= (x[:, :, 2].min(1) <= pole_pos[2]).astype(jnp.float32)
        reward *= (x[:, :, 2].max(1) >= pole_pos[2]).astype(jnp.float32)

        reward = reward.mean()

        metrics = {
            'reward': reward,
            'loss': loss_val,
            'speed/sps': sps,
            'speed/training_walltime': training_walltime,
            'speed/timestamp': training_walltime,
            'train/grad_norm': jnp.mean(summary.get('grad_norm', 0)),
            'train/params_norm': jnp.mean(summary.get('params_norm', 0)),
        }

        logging.info(metrics)
        if progress_fn:
            progress_fn(it, metrics)

        tf.summary.scalar('eval_episode_loss', data=np.array(loss_val),
                          step=it * args.num_envs * args.ep_len)
        tf.summary.scalar('eval_episode_reward', data=np.array(reward),
                          step=it * args.num_envs * args.ep_len)

    """
    Training functions
    """

    @partial(custom_vjp, nondiff_argnums=(0,))
    def memory_profiler(num_steps, x):
        return x

    def memory_profiler_fwd(num_steps, x):
        return x, ()

    def memory_profiler_bwd(num_steps, _, g):

        jax.profiler.save_device_memory_profile(f'memory_{num_steps}.prof')
        return (g,)

    def do_one_step(carry, step_index):
        state, params, key = carry
        key, key_sample = jax.random.split(key)
        combo_state = group_state(state)
        logits = policy_model.apply(params, combo_state)
        actions = parametric_action_distribution.sample(logits, key_sample)
        nstate = step_fn(actions, *state)

        if truncation_length is not None and truncation_length > 0:
            nstate = jax.lax.cond(
                jnp.mod(step_index + 1, truncation_length) == 0.,
                jax.lax.stop_gradient, lambda x: x, nstate)

        return (nstate, params, key), (nstate, logits, actions)

    def loss(params, state, key):
        _, (state_list, logit_list, action_list) = jax.lax.scan(
            do_one_step, (state, params, key),
            (jnp.array(range(episode_length // action_repeat))),
            length=episode_length // action_repeat)

        combo_state = group_state(state_list)

        cf_state_loss, cf_gripper_loss, entropy_loss, cf_action_loss = 0, 0, 0, 0

        pred = combo_state.swapaxes(1, 0)[..., None, :].repeat(args.ep_len, -2)
        pred_demo = demo_traj_raw.swapaxes(1, 0)[..., None, :, :].repeat(args.ep_len, -3)
        pred_dis = ((pred - pred_demo) ** 2).mean(-1)  # (batch, 128, 128), distance
        cf_state_loss = (pred_dis.min(-1) * reverse_discounts).mean() * args.deviation_factor

        # for every state in demo_traj_ find closest state in rollout_traj
        demo = demo_traj_raw.swapaxes(1, 0)[..., None, :].repeat(args.ep_len, -2)
        demo_pred = combo_state.swapaxes(1, 0)[..., None, :, :].repeat(args.ep_len, -3)
        demo_dis = ((demo - demo_pred) ** 2).mean(-1)  # (batch, 128, 128), distance
        cf_state_loss += (demo_dis.min(-1) * reverse_discounts).mean()

        # calc action cf loss
        pred_action = action_list.swapaxes(1, 0)[..., None, :].repeat(args.ep_len, -2)
        pred_demo_action = demo_traj_action.swapaxes(1, 0)[..., None, :, :].repeat(args.ep_len, -3)
        pred_dis_action = ((pred_action - pred_demo_action) ** 2).mean(-1)  # (batch, 128, 128), distance
        cf_action_loss = (pred_dis_action.min(-1) * reverse_discounts).mean()

        demo_action = demo_traj_action.swapaxes(1, 0)[..., None, :].repeat(args.ep_len, -2)
        demo_pred_action = action_list.swapaxes(1, 0)[..., None, :, :].repeat(args.ep_len, -3)
        demo_dis_action = ((demo_action - demo_pred_action) ** 2).mean(-1)  # (batch, 128, 128), distance
        cf_action_loss += (demo_dis_action.min(-1) * reverse_discounts).mean()

        # entropy cost
        loc, scale = jnp.split(logit_list, 2, axis=-1)
        sigma_list = jax.nn.softplus(scale) + parametric_action_distribution._min_std
        entropy_loss = -1 * 0.5 * jnp.log(2 * jnp.pi * sigma_list ** 2)
        entropy_loss = entropy_loss.mean(-1).mean()

        final_loss = cf_state_loss + cf_gripper_loss + entropy_loss * args.entropy_factor + cf_action_loss * args.action_cf_factor

        return final_loss, (cf_state_loss, cf_gripper_loss, entropy_loss, cf_action_loss)

    def _minimize(training_state, state):

        def minimize_step(carry, step_idx):
            policy_params, state, key = carry
            grad_raw, (cf_state_loss, cf_gripper_loss, entropy_loss, cf_action_loss) \
                = loss_grad(policy_params, state, key + step_idx.astype(jnp.uint32))

            return carry, (grad_raw, cf_state_loss, cf_gripper_loss, entropy_loss, cf_action_loss)

        _, (grad_raw, cf_state_loss, cf_gripper_loss, entropy_loss, cf_action_loss) = jax.lax.scan(
            minimize_step, (training_state.policy_params, state, training_state.key),
            jnp.array(range(args.vp)), length=args.vp)

        grad_raw = jax.tree_multimap(lambda t: jnp.nan_to_num(t), grad_raw)
        grad = clip_by_global_norm(grad_raw)
        grad = jax.tree_multimap(lambda t: t.mean(0), grad)

        params_update, optimizer_state = optimizer.update(grad, training_state.optimizer_state)
        policy_params = optax.apply_updates(training_state.policy_params, params_update)

        metrics = {
            'grad_norm': optax.global_norm(grad_raw),
            'params_norm': optax.global_norm(policy_params),
            'cf_state_loss': cf_state_loss.mean(0),
            "cf_gripper_loss": cf_gripper_loss.mean(0),
            'entropy_loss': entropy_loss.mean(0),
            "cf_action_loss": cf_action_loss.mean(0)
        }
        return TrainingState(
            key=key,
            optimizer_state=optimizer_state,
            il_optimizer_state=training_state.il_optimizer_state,
            policy_params=policy_params), metrics

    def clip_by_global_norm(updates):
        g_norm = optax.global_norm(updates)
        trigger = g_norm < max_gradient_norm
        updates = jax.tree_multimap(
            lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
            updates)

        return updates

    # compile training functions
    il_loss_grad = jax.grad(il_loss, has_aux=True)
    loss_grad = jax.grad(loss, has_aux=True)
    _minimize = jax.jit(_minimize)
    il_minimize = jax.jit(il_minimize)
    memory_profiler.defvjp(memory_profiler_fwd, memory_profiler_bwd)

    # prepare training
    sps = 0
    training_walltime = 0
    summary = {'params_norm': optax.global_norm(jax.tree_map(lambda x: x[0], policy_params))}
    training_state = TrainingState(key=key, optimizer_state=optimizer_state,
                                   il_optimizer_state=il_optimizer_state,
                                   policy_params=policy_params)

    # IL bootstrap
    if args.il:
        for it in range(2000):
            # il optimization
            training_state, summary, loss_val = il_minimize(training_state)
            print('IL bootstrap starting iteration %s %s', it, np.array(loss_val), time.time() - xt)

    # main training loop
    for it in range(log_frequency + 1):
        logging.info('starting iteration %s %s', it, time.time() - xt)
        t = time.time()

        if it % 5 == 0:
            visualize(training_state.policy_params, visualize_first_state, key_debug)
            eval_policy(it, key_debug)

        if it == log_frequency:
            break

        # optimization
        t = time.time()
        num_steps = it * args.num_envs * args.ep_len
        training_state, metrics = _minimize(training_state, first_state)

        tf.summary.scalar('cf_state_loss', data=np.array(metrics['cf_state_loss']), step=num_steps)
        tf.summary.scalar('cf_gripper_loss', data=np.array(metrics['cf_gripper_loss']), step=num_steps)
        tf.summary.scalar('entropy_loss', data=np.array(metrics['entropy_loss']), step=num_steps)
        tf.summary.scalar('cf_action_loss', data=np.array(metrics['cf_action_loss']), step=num_steps)
        tf.summary.scalar('grad_norm', data=np.array(metrics['grad_norm']), step=num_steps)
        tf.summary.scalar('params_norm', data=np.array(metrics['params_norm']), step=num_steps)
        print("cf_state_loss", np.array(metrics['cf_state_loss']),
              "cf_action_loss", np.array(metrics['cf_action_loss']),
              "entropy_loss", np.array(metrics['entropy_loss']),
              "grad_norm", np.array(metrics['grad_norm']))

        sps = (episode_length * num_envs) / (time.time() - t)
        training_walltime += time.time() - t

    params = jax.tree_map(lambda x: x[0], training_state.policy_params)
    normalizer_params = jax.tree_map(lambda x: x[0],
                                     training_state.normalizer_params)
    params = normalizer_params, params
    inference = make_inference_fn(core_env.observation_size, core_env.action_size,
                                  normalize_observations)


def make_direct_optimization_model(parametric_action_distribution, obs_size):

    return networks.make_model(
        [512, 256, parametric_action_distribution.param_size],
        obs_size,
        activation=linen.swish)


def make_inference_fn(observation_size, action_size, normalize_observations):
    """Creates params and inference function for the direct optimization agent."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size)
    _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
        observation_size, normalize_observations)
    policy_model = make_direct_optimization_model(parametric_action_distribution,
                                                  observation_size)

    def inference_fn(params, obs, key):
        normalizer_params, params = params
        obs = obs_normalizer_apply_fn(normalizer_params, obs)
        action = parametric_action_distribution.sample(
            policy_model.apply(params, obs), key)
        return action

    return inference_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default="hang_cloth")
    parser.add_argument('--ep_len', default=80, type=int)
    parser.add_argument('--num_envs', default=10, type=int)
    parser.add_argument('--vp', default=5, type=int, help="virtual p size")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--trunc_len', default=10, type=int)
    parser.add_argument('--max_it', default=3000, type=int)
    parser.add_argument('--max_grad_norm', default=0.3, type=float)
    parser.add_argument('--reverse_discount', default=1.0, type=float)
    parser.add_argument('--entropy_factor', default=0, type=float)
    parser.add_argument('--deviation_factor', default=1.0, type=float)
    parser.add_argument('--action_cf_factor', default=0, type=float)
    parser.add_argument('--il', default=1, type=float)
    parser.add_argument('--seed', default=1, type=int)

    args = parser.parse_args()

    envs = {
        "hang_cloth": make_env_hang_cloth
    }

    train(environment_fn=envs[args.env],
          episode_length=args.ep_len,
          num_envs=args.num_envs,
          num_eval_envs=128,
          learning_rate=args.lr,
          normalize_observations=True,
          log_frequency=args.max_it,
          truncation_length=args.trunc_len,
          max_gradient_norm=args.max_grad_norm,
          seed=args.seed)
