# Kubric from scratch

Blender now has `bpy` in PyPI, which makes installation of Kubric easy.
To install the environment using Conda, do this:

```bash
CONDA_ENV_NAME=kubric
conda create -y --name ${CONDA_ENV_NAME} python=3.11
conda activate ${CONDA_ENV_NAME}
conda install -y 'numpy<2.0.0' matplotlib scikit-learn absl-py imageio pypng traitlets importlib_resources munch
conda install -y -c conda-forge pyquaternion openexr trimesh pybullet
pip install OpenEXR
pip install 'etils[epath_no_tf]'
pip install tensorflow tensorflow-datasets tensorflow-graphics
pip install mediapy
pip install --no-cache-dir bpy
pip install apache-beam
pip install -U Pillow
```

To remove the environment and start from scratch, use:

```bash
conda remove -y -n ${CONDA_ENV_NAME} --all
```

To install Kubric code (with small fixes for recent versions of the libraries):

```bash
git clone git@github.com:vedaldi/kubric.git kubric
```

To test it

```bash
python kubric/examples/helloworld.py
```
