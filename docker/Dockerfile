ARG cuda_version=10.2
FROM nvidia/cuda:${cuda_version}-cudnn8-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG cuda_version

# Use bash to support string substitution.
SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-${cuda_version/./-} \
      git \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda package manager.
ENV conda_home=/opt/conda
RUN wget -q -P /tmp \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p ${conda_home} \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

ENV profold_home=/app/profold2
COPY . ${profold_home}
RUN wget -q -P ${profold_home}/profold2/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Install conda and pip packages.
ENV PATH="${conda_home}/bin:${PATH}"
RUN conda update -qy conda \
    && conda install python=3.7 \
    && conda install pytorch torchvision torchaudio cudatoolkit=${cuda_version} -c pytorch \
    && pip install --upgrade pip \
    && pip install -r ${profold_home}/requirements.txt

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.
WORKDIR /app/profold2
RUN echo -e "#!/bin/bash\n\
cd /app/profold2\n\
ldconfig\n\
runner=\"trainer\"\n\
if [ -n \"\$1\" ]; then\n\
  runner=\"\$1\"\n\
  shift\n\
fi\n\
if [ x\"\${runner}\" != x\"trainer\" -a x\"\${runner}\" != x\"evaluator\" ]; then\n\
  echo \"usage: run_profold.sh <trainer|evaluator> <args>\"\n\
  exit 1\n\
fi\n\
python \${runner}.py --torch_home=${profold_home}/.cache/torch --scn_dir=${profold_home}/.cache/sidechainnet_data \$@" > /app/run_profold.sh \
    && chmod +x /app/run_profold.sh
ENTRYPOINT ["/app/run_profold.sh"]
