# Installation
- clone this repo: 
```
https://github.com/SergioMOrozco/decision_making_project.git
cd decision_making_project
```
## Install Dependencies
- This project requires the use of Python 3.11, which can be installed using Conda like so:
```
conda create -n dreamer_v3 python=3.11
conda activate dreamer_v3
```
- install JAX for NVIDIA GPU hardware:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

- install Gym dependencies:
```
pip install setuptools==65.5.0 pip==21
```
- install Dreamer dependencies:
```
cd dreamerv3/
pip install -r requirements.txt
```
## Confirm Installation
- You can confirm that dreamerV3 was installed correctly by running the following without error:
```
python example.py
```
