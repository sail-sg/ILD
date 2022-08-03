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

import argparse
import pickle
import time
from functools import partial
from typing import Any, Callable, Dict, Optional

import brax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import streamlit.components.v1 as components
import tensorflow as tf
from absl import logging
from brax import envs
from brax.io import html
from brax.training import distribution
from brax.training import networks
from brax.training import normalization
from brax.training import pmap
from brax.training.types import PRNGKey
from brax.training.types import Params
from flax import linen
from jax import custom_vjp

logging.set_verbosity(logging.INFO)
tf.config.experimental.set_visible_devices([], "GPU")


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    key: PRNGKey
    normalizer_params: Params
    state_normalizer_params: Params
    optimizer_state: optax.OptState
    il_optimizer_state: optax.OptState
    policy_params: Params


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
    args.logdir = f"multi_traj_logs/{args.env}/{args.env}_ep_len{args.ep_len}_num_envs{args.num_envs}_lr{args.lr}_trunc_len{args.trunc_len}" \
                  f"_max_it{args.max_it}_max_grad_norm{args.max_grad_norm}_ef_{args.entropy_factor}" \
                  f"_df_{args.deviation_factor}_acf_{args.action_cf_factor}_l2loss_{args.l2}_il_{args.il}_ILD_{args.ILD}" \
                  f"/seed{args.seed}"
    demo_traj = np.load(f"expert_multi_traj/{args.env}_traj_state.npy")
    demo_traj = jnp.array(demo_traj)[:args.ep_len]
    demo_traj_action = np.load(f"expert_multi_traj/{args.env}_traj_action.npy")
    demo_traj_action = jnp.array(demo_traj_action)[:args.ep_len]
    demo_traj_obs = np.load(f"expert_multi_traj/{args.env}_traj_observation.npy")
    demo_traj_obs = jnp.array(demo_traj_obs)[:args.ep_len]

    demo_traj_reward = np.load(f"expert_multi_traj/{args.env}_traj_reward.npy")
    print("expert reward", demo_traj_reward, "avg", demo_traj_reward.mean())

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
    key_env = jax.random.split(key_env, process_count)[process_id]
    key = jax.random.split(key, process_count)[process_id]
    key_debug = jax.random.PRNGKey(seed + 666)

    # envs
    core_env = environment_fn(
        action_repeat=action_repeat,
        batch_size=num_envs // local_devices_to_use // process_count,
        episode_length=episode_length)
    key_envs = jax.random.split(key_env, local_devices_to_use)
    step_fn = jax.jit(core_env.step)
    reset_fn = jax.jit(jax.vmap(core_env.reset))
    first_state = reset_fn(key_envs)

    eval_env = environment_fn(
        action_repeat=action_repeat,
        batch_size=num_eval_envs,
        episode_length=episode_length,
        eval_metrics=True)
    eval_step_fn = jax.jit(eval_env.step)
    eval_first_state = jax.jit(eval_env.reset)(key_env)

    # initialize policy
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=core_env.action_size)
    policy_model = make_direct_optimization_model(parametric_action_distribution, core_env.observation_size)

    # init optimizer
    policy_params = policy_model.init(key_models)
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(policy_params)
    il_optimizer_state = optimizer.init(policy_params)
    optimizer_state, policy_params, il_optimizer_state = pmap.bcast_local_devices(
        (optimizer_state, policy_params, il_optimizer_state), local_devices_to_use)

    # observation normalizer
    normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
        normalization.create_observation_normalizer(
            core_env.observation_size,
            normalize_observations,
            num_leading_batch_dims=2,
            pmap_to_devices=local_devices_to_use))

    # state normalizer
    state_normalizer_params, state_normalizer_update_fn, state_normalizer_apply_fn = (
        normalization.create_observation_normalizer(
            demo_traj.shape[-1],
            normalize_observations=True,
            num_leading_batch_dims=2,
            pmap_to_devices=local_devices_to_use))

    """
    IL boostrap
    """

    def il_loss(params, normalizer_params, key):

        normalizer_params = obs_normalizer_update_fn(normalizer_params, demo_traj_obs)
        normalized_obs = obs_normalizer_apply_fn(normalizer_params, demo_traj_obs)
        logits = policy_model.apply(params, normalized_obs)
        rollout_actions = parametric_action_distribution.sample(logits, key)

        loss_val = (rollout_actions - demo_traj_action) ** 2
        loss_val = loss_val.sum(-1).mean()
        return loss_val, normalizer_params

    def il_minimize(training_state: TrainingState):
        synchro = pmap.is_replicated((training_state.optimizer_state,
                                      training_state.policy_params,
                                      training_state.normalizer_params,
                                      training_state.state_normalizer_params,
                                      training_state.il_optimizer_state), axis_name='i')
        key, key_grad = jax.random.split(training_state.key)

        grad, normalizer_params = il_loss_grad(training_state.policy_params,
                                               training_state.normalizer_params,
                                               key_grad)
        grad = clip_by_global_norm(grad)
        grad = jax.lax.pmean(grad, axis_name='i')
        params_update, il_optimizer_state = optimizer.update(grad, training_state.il_optimizer_state)
        policy_params = optax.apply_updates(training_state.policy_params, params_update)

        metrics = {
            'grad_norm': optax.global_norm(grad),
            'params_norm': optax.global_norm(policy_params)
        }
        return TrainingState(
            key=key,
            optimizer_state=training_state.optimizer_state,
            il_optimizer_state=il_optimizer_state,
            normalizer_params=normalizer_params,
            state_normalizer_params=training_state.state_normalizer_params,
            policy_params=policy_params), metrics, synchro

    """
    Evaluation functions
    """

    def do_one_step_eval(carry, unused_target_t):
        state, params, normalizer_params, key = carry
        key, key_sample = jax.random.split(key)
        # TODO: Make this nicer ([0] comes from pmapping).
        obs = obs_normalizer_apply_fn(
            jax.tree_map(lambda x: x[0], normalizer_params), state.obs)
        print(obs.shape)
        print(jax.tree_map(lambda x: x.shape, params))
        logits = policy_model.apply(params, obs)
        actions = parametric_action_distribution.sample(logits, key_sample)
        nstate = eval_step_fn(state, actions)
        return (nstate, params, normalizer_params, key), state

    @jax.jit
    def run_eval(params, state, normalizer_params, key):
        params = jax.tree_map(lambda x: x[0], params)
        (state, _, _, key), state_list = jax.lax.scan(
            do_one_step_eval, (state, params, normalizer_params, key), (),
            length=episode_length // action_repeat)
        return state, key, state_list

    def eval_policy(it, key_debug):
        if process_id == 0:
            eval_state, key_debug, state_list = run_eval(training_state.policy_params,
                                                         eval_first_state,
                                                         training_state.normalizer_params,
                                                         key_debug)
            eval_metrics = eval_state.info['eval_metrics']
            eval_metrics.completed_episodes.block_until_ready()
            eval_sps = (
                    episode_length * eval_first_state.reward.shape[0] /
                    (time.time() - t))
            avg_episode_length = (
                    eval_metrics.completed_episodes_steps /
                    eval_metrics.completed_episodes)
            metrics = dict(
                dict({
                    f'eval/episode_{name}': value / eval_metrics.completed_episodes
                    for name, value in eval_metrics.completed_episodes_metrics.items()
                }),
                **dict({
                    'eval/completed_episodes': eval_metrics.completed_episodes,
                    'eval/avg_episode_length': avg_episode_length,
                    'speed/sps': sps,
                    'speed/eval_sps': eval_sps,
                    'speed/training_walltime': training_walltime,
                    'speed/timestamp': training_walltime,
                    'train/grad_norm': jnp.mean(summary.get('grad_norm', 0)),
                    'train/params_norm': jnp.mean(summary.get('params_norm', 0)),
                }))

            logging.info(metrics)
            if progress_fn:
                progress_fn(it, metrics)

            if it % 10 == 0:
                visualize(state_list)

            tf.summary.scalar('eval_episode_reward', data=np.array(metrics['eval/episode_reward']),
                              step=it * args.num_envs * args.ep_len)

    """
    Training functions
    """

    @partial(custom_vjp)
    def norm_grad(x):
        return x

    def norm_grad_fwd(x):
        return x, ()

    def norm_grad_bwd(x, g):
        # g /= jnp.linalg.norm(g)
        # g = jnp.nan_to_num(g)
        g_norm = optax.global_norm(g)
        trigger = g_norm < 1.0
        g = jax.tree_multimap(
            lambda t: jnp.where(trigger,
                                jnp.nan_to_num(t),
                                (jnp.nan_to_num(t) / g_norm) * 1.0), g)
        return g,

    norm_grad.defvjp(norm_grad_fwd, norm_grad_bwd)

    def do_one_step(carry, step_index):
        state, params, normalizer_params, key = carry
        key, key_sample = jax.random.split(key)
        normalized_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
        logits = policy_model.apply(params, normalized_obs)
        actions = parametric_action_distribution.sample(logits, key_sample)
        nstate = step_fn(state, actions)

        actions = norm_grad(actions)
        nstate = norm_grad(nstate)

        if truncation_length is not None and truncation_length > 0:
            nstate = jax.lax.cond(
                jnp.mod(step_index + 1, truncation_length) == 0.,
                jax.lax.stop_gradient, lambda x: x, nstate)

        return (nstate, params, normalizer_params, key), (nstate.reward, state.obs, state.qp, logits, actions)

    def l2_loss(params, normalizer_params, state_normalizer_params, state, key):
        _, (rewards, obs, qp_list, logit_list, action_list) = jax.lax.scan(
            do_one_step, (state, params, normalizer_params, key),
            (jnp.array(range(episode_length // action_repeat))),
            length=episode_length // action_repeat)

        rollout_traj = jnp.concatenate([qp_list.pos.reshape((qp_list.pos.shape[0], qp_list.pos.shape[1], -1)),
                                        qp_list.rot.reshape((qp_list.rot.shape[0], qp_list.rot.shape[1], -1)),
                                        qp_list.vel.reshape((qp_list.vel.shape[0], qp_list.vel.shape[1], -1)),
                                        qp_list.ang.reshape((qp_list.ang.shape[0], qp_list.ang.shape[1], -1))], axis=-1)

        # normalize states
        normalizer_params = obs_normalizer_update_fn(normalizer_params, obs)
        state_normalizer_params = state_normalizer_update_fn(state_normalizer_params, rollout_traj)

        rollout_traj = state_normalizer_apply_fn(state_normalizer_params, rollout_traj)
        demo_traj_ = state_normalizer_apply_fn(state_normalizer_params, demo_traj)

        loss_val = (rollout_traj - demo_traj_) ** 2
        loss_val = jnp.sqrt(loss_val.sum(-1)).mean()

        return loss_val, (normalizer_params, state_normalizer_params, obs, 0, 0, 0)

    def loss(params, normalizer_params, state_normalizer_params, state, key):
        _, (rewards, obs, qp_list, logit_list, action_list) = jax.lax.scan(
            do_one_step, (state, params, normalizer_params, key),
            (jnp.array(range(episode_length // action_repeat))),
            length=episode_length // action_repeat)

        rollout_traj_raw = jnp.concatenate([qp_list.pos.reshape((qp_list.pos.shape[0], qp_list.pos.shape[1], -1)),
                                            qp_list.rot.reshape((qp_list.rot.shape[0], qp_list.rot.shape[1], -1)),
                                            qp_list.vel.reshape((qp_list.vel.shape[0], qp_list.vel.shape[1], -1)),
                                            qp_list.ang.reshape((qp_list.ang.shape[0], qp_list.ang.shape[1], -1))],
                                           axis=-1)

        # normalize states
        normalizer_params = obs_normalizer_update_fn(normalizer_params, obs)
        state_normalizer_params = state_normalizer_update_fn(state_normalizer_params, rollout_traj_raw)
        rollout_traj_raw = state_normalizer_apply_fn(state_normalizer_params, rollout_traj_raw)

        # (num_envs,num_demo,num_step,features) (360,16,128,130)
        rollout_traj = rollout_traj_raw.swapaxes(1, 0)[:, None, ...].repeat(demo_traj.shape[1], 1)
        demo_traj_ = state_normalizer_apply_fn(state_normalizer_params, demo_traj)
        demo_traj_ = demo_traj_.swapaxes(1, 0)[None, ...].repeat(args.num_envs, 0)

        # calc state chamfer loss
        # for every state in rollout_traj find closest state in demo
        pred = rollout_traj[..., None, :].repeat(args.ep_len, -2)
        pred_demo = demo_traj_[..., None, :, :].repeat(args.ep_len, -3)
        pred_dis = jnp.sqrt(((pred - pred_demo) ** 2).mean(-1)).min(-1)  # (360,16, 128), distance
        cf_loss = pred_dis.mean(-1).min(-1).mean() * args.deviation_factor  # select the best from k expert demos

        # for every state in demo_traj_ find closest state in rollout_traj
        demo = demo_traj_[..., None, :].repeat(args.ep_len, -2)
        demo_pred = rollout_traj[..., None, :, :].repeat(args.ep_len, -3)
        demo_dis = jnp.sqrt(((demo - demo_pred) ** 2).mean(-1)).min(-1)  # (batch, 128, 128), distance
        cf_loss += demo_dis.mean(-1).min(-1).mean()

        cf_action_loss, entropy_loss = 0, 0
        final_loss = cf_loss + entropy_loss * args.entropy_factor + cf_action_loss * args.action_cf_factor
        final_loss = jnp.tanh(final_loss)

        return final_loss, (normalizer_params, state_normalizer_params, obs, cf_loss, entropy_loss, cf_action_loss)

    def _minimize(training_state: TrainingState, state: envs.State):
        synchro = pmap.is_replicated((training_state.optimizer_state,
                                      training_state.policy_params,
                                      training_state.normalizer_params,
                                      training_state.state_normalizer_params), axis_name='i')
        key, key_grad = jax.random.split(training_state.key)
        grad_raw, (normalizer_params,
                   state_normalizer_params,
                   obs, cf_loss, entropy_loss, cf_action_loss) = loss_grad(training_state.policy_params,
                                                                           training_state.normalizer_params,
                                                                           training_state.state_normalizer_params,
                                                                           state, key_grad)
        grad_raw = jax.tree_multimap(lambda t: jnp.nan_to_num(t), grad_raw)
        grad = clip_by_global_norm(grad_raw)
        grad = jax.lax.pmean(grad, axis_name='i')
        params_update, optimizer_state = optimizer.update(grad, training_state.optimizer_state)
        policy_params = optax.apply_updates(training_state.policy_params, params_update)

        metrics = {
            'grad_norm': optax.global_norm(grad_raw),
            'params_norm': optax.global_norm(policy_params),
            'cf_loss': cf_loss,
            'entropy_loss': entropy_loss,
            "cf_action_loss": cf_action_loss
        }
        return TrainingState(
            key=key,
            optimizer_state=optimizer_state,
            il_optimizer_state=training_state.il_optimizer_state,
            normalizer_params=normalizer_params,
            state_normalizer_params=state_normalizer_params,
            policy_params=policy_params), metrics, synchro

    def clip_by_global_norm(updates):
        g_norm = optax.global_norm(updates)
        trigger = g_norm < max_gradient_norm
        updates = jax.tree_multimap(
            lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm),
            updates)

        return updates

    # compile training functions
    il_loss_grad = jax.grad(il_loss, has_aux=True)

    if args.l2 == 1:
        loss_grad = jax.grad(l2_loss, has_aux=True)
        print("using l2 loss")
    else:
        loss_grad = jax.grad(loss, has_aux=True)
        print("using chamfer loss")
    minimize = jax.pmap(_minimize, axis_name='i')
    il_minimize = jax.pmap(il_minimize, axis_name='i')

    # prepare training
    sps = 0
    training_walltime = 0
    summary = {'params_norm': optax.global_norm(jax.tree_map(lambda x: x[0], policy_params))}
    key = jnp.stack(jax.random.split(key, local_devices_to_use))
    training_state = TrainingState(key=key, optimizer_state=optimizer_state,
                                   il_optimizer_state=il_optimizer_state,
                                   normalizer_params=normalizer_params,
                                   state_normalizer_params=state_normalizer_params,
                                   policy_params=policy_params)

    # IL bootstrap
    if args.il:
        for it in range(1000):
            logging.info('IL bootstrap starting iteration %s %s', it, time.time() - xt)
            t = time.time()

            if it % 100 == 0:
                eval_policy(it, key_debug)

            # il optimization
            training_state, summary, synchro = il_minimize(training_state)
            assert synchro[0], (it, training_state)
            jax.tree_map(lambda x: x.block_until_ready(), summary)
        eval_policy(0, key_debug)

    # main training loop
    if args.ILD:
        for it in range(log_frequency + 1):
            actor_lr = (1e-5 - args.lr) * float(it / log_frequency) + args.lr
            optimizer = optax.adam(learning_rate=actor_lr)
            print("actor_lr: ", actor_lr)

            logging.info('starting iteration %s %s', it, time.time() - xt)
            t = time.time()

            eval_policy(it, key_debug)
            if it == log_frequency:
                break

            # optimization
            t = time.time()
            num_steps = it * args.num_envs * args.ep_len
            training_state, metrics, synchro = minimize(training_state, first_state)
            tf.summary.scalar('cf_loss', data=np.array(metrics['cf_loss'])[0], step=num_steps)
            tf.summary.scalar('entropy_loss', data=np.array(metrics['entropy_loss'])[0], step=num_steps)
            tf.summary.scalar('cf_action_loss', data=np.array(metrics['cf_action_loss'])[0], step=num_steps)
            tf.summary.scalar('grad_norm', data=np.array(metrics['grad_norm'])[0], step=num_steps)
            tf.summary.scalar('params_norm', data=np.array(metrics['params_norm'])[0], step=num_steps)
            assert synchro[0], (it, training_state)
            jax.tree_map(lambda x: x.block_until_ready(), metrics)
            sps = (episode_length * num_envs) / (time.time() - t)
            training_walltime += time.time() - t

    params = jax.tree_map(lambda x: x[0], training_state.policy_params)
    normalizer_params = jax.tree_map(lambda x: x[0],
                                     training_state.normalizer_params)
    params = normalizer_params, params
    inference = make_inference_fn(core_env.observation_size, core_env.action_size,
                                  normalize_observations)

    # save params in pickle file
    with open(args.logdir + '/params.pkl', 'wb') as f:
        pickle.dump(params, f)

    pmap.synchronize_hosts()


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


def visualize(state_list):
    environment = args.env
    env = envs.create(env_name=environment)

    visual_states = []
    for i in range(state_list.qp.ang.shape[0]):
        qp_state = brax.QP(np.array(state_list.qp.pos[i, 0]),
                           np.array(state_list.qp.rot[i, 0]),
                           np.array(state_list.qp.vel[i, 0]),
                           np.array(state_list.qp.ang[i, 0]))
        visual_states.append(qp_state)

    html_string = html.render(env.sys, visual_states)
    components.html(html_string, height=500)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default="reacher")
    parser.add_argument('--ep_len', default=128, type=int)
    parser.add_argument('--num_envs', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--trunc_len', default=10, type=int)
    parser.add_argument('--max_it', default=5000, type=int)
    parser.add_argument('--max_grad_norm', default=0.3, type=float)
    parser.add_argument('--entropy_factor', default=0, type=float)
    parser.add_argument('--deviation_factor', default=1.0, type=float)
    parser.add_argument('--action_cf_factor', default=0, type=float)
    parser.add_argument('--il', default=1, type=float)
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--ILD', default=1, type=int)

    args = parser.parse_args()

    train(environment_fn=envs.create_fn(args.env),
          episode_length=args.ep_len,
          num_envs=args.num_envs,
          learning_rate=args.lr,
          normalize_observations=True,
          log_frequency=args.max_it,
          truncation_length=args.trunc_len,
          max_gradient_norm=args.max_grad_norm,
          seed=args.seed)
