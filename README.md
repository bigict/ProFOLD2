# ProFOLD2

![GitHub License](https://img.shields.io/github/license/bigict/ProFOLD2)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/bigict/ProFOLD2/pylint.yml)
![GitHub Release](https://img.shields.io/github/v/release/bigict/ProFOLD2)

ProFOLD2 - A protein 3D-structure prediction application

## Requirements

* [Python3.9+](https://www.python.org)
* [Dependencies](https://github.com/bigict/ProFOLD2/network/dependencies)

## Running ProFOLD2

1.  Clone this repository and `cd` into it.
  ```bash
  $git clone https://github.com/bigict/ProFOLD2.git
  $cd ProFOLD2
  $git submodule update --init  # required if use FusedEvoformer, recommended.
  ```
2.  Create a [virtual enviroment](https://anaconda.org) and install [dependencies](https://github.com/bigict/ProFOLD2/network/dependencies)
  ```bash
  $conda create -n profold2 python=3.11
  $conda activate profold2
  $bash install_env.sh
  ```
3.  Train a model
  ```bash
  $python main.py train --prefix=OUTPUT_DIR
  ```
  
  There are a lot of parameters, you can run
    
  ```bash
  $python main.py train -h
  ```
  
  for further help.
  
  `ProFOLD2` logs it's metrics to [TensorBoard](https://www.tensorflow.org/tensorboard). You can run
  
  ```bash
  $tensorboard --logdir=OUTPUT_DIR
  ```
  
  Then open http://localhost:6006 in you browser.
  
4.  Inference
  ```bash
  $python main.py predict --models [MODEL_NAME1:]MODEL_FILE1 [MODEL_NAME2:]MODEL_FILE2
  ```
  
  Just like `train`, you can run
  ```bash
  $python main.py predict -h
  ```
  
