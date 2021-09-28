# ProFOLD2

[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

ProFOLD2 - A protein 3D structure prediction application

## Requirements

* [Python3.6+](https://www.python.org)
* [Dependencies](https://github.com/bigict/ProFOLD2/network/dependencies)

## Running ProFOLD2

1.  Clone this repository and `cd` into it.
  ```bash
  $git clone https://github.com/bigict/ProFOLD2.git
  ```
2.  Create a [virtual enviroment](https://docs.python.org/3/library/venv.html) and install [dependencies](https://github.com/bigict/ProFOLD2/network/dependencies)
  ```bash
  $python -m venv env
  $./env/bin/pip install -r requirements.txt
  ```
3.  Train a model
  ```bash
  $./env/bin/python trainer.py --prefix=OUTPUT_DIR
  ```
  
  There are a lot of parameters, you can run
    
  ```bash
  $./env/bin/python trainer.py -h
  ```
  
  for further help.
  
  `ProFOLD2` logs it's metrics to [TensorBoard](https://www.tensorflow.org/tensorboard). You can run
  
  ```bash
  $tensorboard --logdir=OUTPUT_DIR
  ```
  
  Then open http://localhost:6006 in you browser.
  
4.  Inference
  ```bash
  $./env/bin/python evaluator.py --model=MODEL_FILE
  ```
  
  Just like `trainer.py`, you can run
  ```bash
  $./env/bin/python evaluator.py -h
  ```
  
