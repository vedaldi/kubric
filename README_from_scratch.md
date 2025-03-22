# Kubric from scratch

Blender now has `bpy` in PyPI, which makes installation of Kubric easy.
To install the environment using Conda, do this:

```bash
CONDA_ENV_NAME=kubric
conda create -y --name ${CONDA_ENV_NAME} python=3.11
conda activate ${CONDA_ENV_NAME}
conda install -y -c conda-forge absl-py etils imageio importlib_resources munch 'numpy<2.0.0' pypng pyquaternion tensorflow traitlets openexr scikit-learn trimesh apache-beam
pip install --no-cache-dir OpenEXR bpy
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
