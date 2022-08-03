import logging
import os
import pickle

import jax
import numpy as np

from brax import envs
from brax.training import ppo, sac

my_path = os.path.dirname(os.path.abspath(__file__))
logging.getLogger().setLevel(logging.INFO)

def train_ppo(env_name, algo="ppo"):
    if algo == "ppo":
        inference, params, metrics = ppo.train(
            envs.create_fn(env_name),
            num_timesteps=1e7,
            episode_length=128,
            num_envs=64,
            learning_rate=3e-4,
            entropy_cost=1e-2,
            discounting=0.95,
            unroll_length=5,
            batch_size=64,
            num_minibatches=8,
            num_update_epochs=4,
            log_frequency=5,
            normalize_observations=True,
            reward_scaling=10)
    else:
        inference, params, metrics = sac.train(
            envs.create_fn(env_name),
            num_timesteps=5000000,
            episode_length=128,
            num_envs=4,
            learning_rate=3e-4,
            discounting=0.99,
            batch_size=4,
            log_frequency=10000,
            normalize_observations=True,
            reward_scaling=0.1,
            min_replay_size=10000,
            max_replay_size=100000,
            grad_updates_per_step=64,)

    # save paras into pickle
    with open(f"{my_path}/../brax_task/expert_multi_traj/{env_name}_params.pickle", "wb") as f:
        pickle.dump(params, f)


def rollout(env_name, num_steps, num_envs):
    env_fn = envs.create_fn(env_name)
    env = env_fn(batch_size=num_envs * 10, episode_length=num_steps * 2)
    env.step = jax.jit(env.step)

    inference = ppo.make_inference_fn(env.observation_size, env.action_size, True)
    inference = jax.jit(inference)
    # load pickle file
    with open(f"{my_path}/../brax_task/expert_multi_traj/{env_name}_params.pickle", "rb") as f:
        decoded_params = pickle.load(f)

    traj_states = []
    traj_actions = []
    traj_obs = []
    traj_rewards = 0
    traj_done = 0

    state = env.reset(jax.random.PRNGKey(0))
    for j in range(num_steps):
        print(env_name, "step: ", j)
        action = inference(decoded_params, state.obs, jax.random.PRNGKey(0))
        state = env.step(state, action)
        qp = np.concatenate([state.qp.pos.reshape((state.qp.pos.shape[0], -1)),
                             state.qp.rot.reshape((state.qp.rot.shape[0], -1)),
                             state.qp.vel.reshape((state.qp.vel.shape[0], -1)),
                             state.qp.ang.reshape((state.qp.ang.shape[0], -1))], axis=-1)

        traj_states.append(qp)
        traj_actions.append(action)
        traj_obs.append(state.obs)
        traj_rewards += state.reward
        traj_done += state.done

    os.makedirs(f"{my_path}/../brax_task/expert_multi_traj", exist_ok=True)
    print(env_name, "traj reward: ", traj_rewards)
    print(env_name, "traj done: ", traj_done)

    traj_rewards = np.array(traj_rewards)
    traj_states = np.array(traj_states)
    traj_actions = np.array(traj_actions)
    traj_obs = np.array(traj_obs)
    traj_done = np.array(traj_done)

    # assert traj_done.sum() <= 2
    # filter by traj done
    traj_states = traj_states[:, traj_done == 0]
    traj_actions = traj_actions[:, traj_done == 0]
    traj_obs = traj_obs[:, traj_done == 0]
    traj_rewards = traj_rewards[traj_done == 0]

    # get_idx from top k rewards
    top_k_idx = np.argsort(traj_rewards)[-num_envs:]
    print(env_name, "top k rewards: ", traj_rewards[top_k_idx])

    np.save(f"{my_path}/../brax_task/expert_multi_traj/%s_traj_state.npy" % env_name, traj_states[:, top_k_idx])
    np.save(f"{my_path}/../brax_task/expert_multi_traj/%s_traj_action.npy" % env_name, traj_actions[:, top_k_idx])
    np.save(f"{my_path}/../brax_task/expert_multi_traj/%s_traj_observation.npy" % env_name, traj_obs[:, top_k_idx])
    np.save(f"{my_path}/../brax_task/expert_multi_traj/%s_traj_reward.npy" % env_name, traj_rewards[top_k_idx])

    np.save(f"{my_path}/../brax_task/expert/%s_traj_state.npy" % env_name, traj_states[:, top_k_idx[-1]])
    np.save(f"{my_path}/../brax_task/expert/%s_traj_action.npy" % env_name, traj_actions[:, top_k_idx[-1]])
    np.save(f"{my_path}/../brax_task/expert/%s_traj_obs.npy" % env_name, traj_obs[:, top_k_idx[-1]])
    np.save(f"{my_path}/../brax_task/expert/%s_traj_reward.npy" % env_name, traj_rewards[top_k_idx[-1]])
    return traj_states, traj_actions, traj_obs


def print_demonstration_reward():
    # Ant & Hopper & Humanoid & Reacher & Walker2d & Swimmer & Inverted pendulum & Acrobot
    env_names = ["ant", "hopper", "humanoid", "reacher", "walker2d", "swimmer", "inverted_pendulum", "acrobot"]
    line = ""
    for env_name in env_names:
        print(env_name)
        traj_rewards = np.load(f"{my_path}/../brax_task/expert_multi_traj/{env_name}_traj_reward.npy")
        line += f" & {traj_rewards.mean():.2f} $\pm$ {traj_rewards.std():.2f}"
    print(line)


if __name__ == '__main__':
    # print_demonstration_reward()
    train_ppo("humanoid", algo="sac")
    rollout("humanoid", num_steps=128, num_envs=16)
    # env_names = ["ant", "walker2d", "humanoid", "acrobot", "reacher", "hopper", "swimmer", "inverted_pendulum"]
    # for env_name in env_names:
    #     rollout(env_name, num_steps=128, num_envs=16)
