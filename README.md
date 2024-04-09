# Installation
- clone this repo: 
```
https://github.com/SergioMOrozco/decision_making_project.git
cd decision_making_project
```
## Create Environment
- create environment and install softgym dependencies 
```
conda create -n softgym python=3.11
conda activate softgym 
```

## Install PyFleX For Softgym
- install docker-ce, then:
```
sudo docker pull xingyu/softgym
```

- install pybind11 like so:
```
conda install pybind11
```

- run docker container like so (obviously change to where PyFleX and anacodna is installed on your machine):
```
sudo docker run -v ~/dev/decision_making_project/softagent/softgym/:/workspace/softgym/ -v ~/anaconda3/envs/softgym/:/workspace/anaconda3/envs/softgym/ -v /tmp/.X11-unix/:/tmp/.X11-unix/ --gpus all -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -it xingyu/softgym:latest bash

```

- Compile PyFleX in docker environment:
```
export PATH="/workspace/anaconda3/envs/softgym/bin:$PATH"
cd softgym
. ./prepare_1.0.sh && ./compile_1.0.sh
```

- exit the docker container, and add the following to ~/.bashrc:
```
export PYFLEXROOT=/home/sorozco0612/dev/decision_making_project/softagent/softgym/PyFleX
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
```

- source ~/.basrhc like so:
```
source ~/.bashrc
conda activate softgym
```

- confirm that softgym can run with PyFlex by running the following example:

```
cd softagent/softgym/
python examples/random_env.py --env_name PassWater
```

- keep installing missing dependencies as they come up. Rather than working with ```environment.yml``` we let pip and conda automatically resolve dependencies and versions

## Install Dreamer Dependencies

- install JAX for NVIDIA GPU hardware:
```
conda activate softgym
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

- install Gym dependencies:
```
python -m pip install --upgrade pip setuptools wheel
```
- install Dreamer dependencies:
```
cd softagent/dreamerv3/
pip install -r requirements.txt
```


## Confirm Installation
- You can confirm that dreamerV3 was installed correctly by running the following without error:
```
python example.py
```

- keep installing missing dependencies as they come up. Rather than working with ```requirements.txt``` we let pip and conda automatically resolve dependencies and versions
