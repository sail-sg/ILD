## Imitation Learning via Differentiable Physics

It recovers expert behavior with a single demonstration via differential dynamics.

#### Installation
~~~
conda create -n ILD python==3.8
conda activate ILD

pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install brax
pip install streamlit
pip install tensorflow
pip install open3d
~~~

#### Start training
~~~
cd policy/brax_task
CUDA_VISIBLE_DEVICES=0 python train_on_policy.py --env="ant" --seed=1

cd policy/cloth_task
CUDA_VISIBLE_DEVICES=0 python train_on_policy.py --seed=1
~~~
