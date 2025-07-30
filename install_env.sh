set -e

help() {
  echo "usage: `basename $0` [-h]"
  echo "options:"
  echo "    -h, --help show this help message and exit"
  echo "    -c, --do-cleanup cleanup cached data"
  exit $1
}

do_cleanup=0

ARGS=$(getopt -o "ch" -l "do-cleanup,help" -- "$@") || help 1
eval "set -- ${ARGS}"
while true; do
  case "$1" in
    (-c | --do-cleanup) do_cleanup=1; shift 1;;
    (-h | --help) help 0 ;;
    (--) shift 1; break;;
    (*) help 1;
  esac
done

cleanup() {
  if [ ${do_cleanup} -ne 0 ]; then
    pip cache purge
    conda clean -a -y -f
  fi
}

cuda_version=${cuda_version:-"12.1.1"}
gcc_version=${gcc_version:-"12.4.0"}
openmm_version=${openmm_version:-"8.0.0"}
pytorch_version=${pytorch_version:-"2.3.1"}
pytorch_cuda=$(echo ${cuda_version}|cut -d. -f1-2)

# conda create -n pf2 python=3.11
# conda activate pf2

conda install -y -c conda-forge \
    gxx=${gcc_version} \
    ninja \
    && cleanup

conda install -y -c conda-forge \
    openmm=${openmm_version} \
    cuda-version=${pytorch_cuda} \
    pdbfixer \
    && cleanup

pip install torch==${pytorch_version} \
    -f https://download.pytorch.org/whl/cu${pytorch_cuda//./} \
    && cleanup

conda install -y -c nvidia \
    cuda-cccl=${pytorch_cuda} \
    cuda-libraries-dev=${pytorch_cuda} \
    cuda-nvcc=${pytorch_cuda} \
    && cleanup

conda install -y -c nvidia/label/cuda-${cuda_version} \
    libcurand-dev \
    && cleanup

conda install -y -c conda-forge \
    biopython \
    biotite \
    einops \
    rdkit \
    tensorboard \
    tqdm \
    && cleanup
