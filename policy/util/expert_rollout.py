import os
import pickle

import jax
import numpy as np
from brax import envs
from brax.training import ppo

my_path = os.path.dirname(os.path.abspath(__file__))


# generate num_traj rollout trajectories from the policy with size of num_steps
# every trajectory is a list of states, actions, observations
# save the states to a file as "expert_multi_traj/$env_name_traj_state.npy"
def rollout(env_name, num_steps, num_envs):
    env_fn = envs.create_fn(env_name)
    env = env_fn(batch_size=num_envs * 2)
    env.step = jax.jit(env.step)

    inference = ppo.make_inference_fn(env.observation_size, env.action_size, True)
    inference = jax.jit(inference)
    # load pickle file
    with open(f"{my_path}/../brax_task/expert/{env_name}_params.pickle", "rb") as f:
        decoded_params = pickle.load(f)

    traj_states = []
    traj_actions = []
    traj_obs = []
    traj_rewards = 0

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

    os.makedirs(f"{my_path}/../brax_task/expert_multi_traj", exist_ok=True)
    print(env_name, "traj reward: ", traj_rewards)

    traj_rewards = np.array(traj_rewards)
    traj_states = np.array(traj_states)
    traj_actions = np.array(traj_actions)
    traj_obs = np.array(traj_obs)

    # get_idx from top k rewards
    top_k_idx = np.argsort(traj_rewards)[-num_envs:]
    print(env_name, "top k rewards: ", traj_rewards[top_k_idx])

    np.save(f"{my_path}/../brax_task/expert_multi_traj/%s_traj_state.npy" % env_name, traj_states[top_k_idx])
    np.save(f"{my_path}/../brax_task/expert_multi_traj/%s_traj_action.npy" % env_name, traj_actions[top_k_idx])
    np.save(f"{my_path}/../brax_task/expert_multi_traj/%s_traj_observation.npy" % env_name, traj_obs[top_k_idx])
    return traj_states, traj_actions, traj_obs


if __name__ == '__main__':
    env_names = ["ant", "walker2d", "humanoid", "acrobot", "reacher", "hopper", "swimmer", "inverted_pendulum"]
    for env_name in env_names:
        rollout(env_name, num_steps=128, num_envs=16)
