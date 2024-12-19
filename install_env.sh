set -e

cuda_version=${cuda_version:-"12.1.1"}
gcc_version=${gcc_version:-"12.4.0"}
openmm_version=${openmm_version:-"8.0.0"}
pytorch_version=${pytorch_version:-"2.3.1"}
pytorch_cuda=$(echo ${cuda_version}|cut -d. -f1-2)

# conda create -n pf2 python=3.11
# conda activate pf2

conda install -y -c conda-forge \
    gxx=${gcc_version} \
    ninja

conda install -y -c conda-forge \
    openmm=${openmm_version} \
    pdbfixer

conda install -y -c pytorch -c nvidia \
    pytorch=${pytorch_version} \
    cuda-cccl=${pytorch_cuda} \
    cuda-libraries-dev=${pytorch_cuda} \
    cuda-nvcc=${pytorch_cuda} \
    pytorch-cuda=${pytorch_cuda}

conda install -y -c nvidia/label/cuda-${cuda_version} \
    libcurand-dev

conda install -y -c conda-forge \
    biopython \
    einops \
    tensorboard
