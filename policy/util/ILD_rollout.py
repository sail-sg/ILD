import os
import pickle

import brax
import jax
import numpy as np
from brax import envs
from brax.io import html
from brax.training import distribution, normalization, ppo
from policy.brax_task.train_on_policy import make_direct_optimization_model
import streamlit.components.v1 as components

my_path = os.path.dirname(os.path.abspath(__file__))


def rollout(env_name, num_steps=128, use_expert=False, seed = 1):
    env_fn = envs.create_fn(env_name)
    env = env_fn(batch_size=1, episode_length=num_steps * 2, auto_reset=False)
    env.step = jax.jit(env.step)

    # initialize policy
    if not use_expert:
        parametric_action_distribution = distribution.NormalTanhDistribution(event_size=env.action_size)
        policy_model = make_direct_optimization_model(parametric_action_distribution, env.observation_size)
        policy_model.apply = jax.jit(policy_model.apply)
    else:
        inference = ppo.make_inference_fn(env.observation_size, env.action_size, True)
        inference = jax.jit(inference)
        with open(f"{my_path}/../brax_task/expert_multi_traj/{env_name}_params.pickle", "rb") as f:
            decoded_params = pickle.load(f)

    with open(f'{env_name}_params.pkl', 'rb') as f:
        normalizer_params, params = pickle.load(f)

    _, _, obs_normalizer_apply_fn = (
        normalization.create_observation_normalizer(
            env.observation_size,
            True,
            num_leading_batch_dims=2,
            pmap_to_devices=1))

    key = jax.random.PRNGKey(seed)
    state = env.reset(jax.random.PRNGKey(seed))

    def do_one_step_eval(carry, unused_target_t):
        state, params, normalizer_params, key = carry
        key, key_sample = jax.random.split(key)

        if not use_expert:
            normalized_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
            logits = policy_model.apply(params, normalized_obs)
            action = parametric_action_distribution.sample(logits, key)
        else:
            action = inference(decoded_params, state.obs, key)

        nstate = env.step(state, action)
        return (nstate, params, normalizer_params, key), state

    _, state_list = jax.lax.scan(
        do_one_step_eval, (state, params, normalizer_params, key), (),
        length=num_steps)

    print(f'{env_name} reward: {state_list.reward.sum():.2f}')
    visualize(state_list, env_name, num_steps)


def visualize(state_list, env_name, num_steps):
    env = envs.create(env_name=env_name, episode_length=num_steps)

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
    rollout("humanoid", num_steps=128, use_expert=False, seed=6)
