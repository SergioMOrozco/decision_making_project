# Installation
- clone this repo: 
```
git clone https://github.com/YunzhuLi/DPI-Net.git
cd DPI-Net
git submodule update --init --recursive
```
## Install Dependencies

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
