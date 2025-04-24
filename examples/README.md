# Multi-node-training on slurm with PyTorch

### What's this?
* A simple note for how to start ProFOLD on [slurm](https://slurm.schedmd.com) scheduler or local machine with [PyTorch](https://pytorch.org)
* Warning: might need to re-factor your own code.

### Run command in cluster
* Training
```
num_batches=1000 sbatch -D . -p <partition> train.job -- --verbose
```

* Evaluate
```
sbatch -D . -p <partition> evaluate.job -- --verbose
```

* Predict
```
sbatch -D . -p <partition> predict.job -- --verbose fasta_file [fasta_file ...]
```

for further help
```
./{train,evaluate,predict}.job -h
```
