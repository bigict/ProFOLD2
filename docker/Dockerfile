ARG CUDA=11.1.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      cuda-command-line-tools-$(echo ${CUDA/./-}|awk -F '.' '{print $1}') \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda package manager.
ENV conda_home=/opt/conda
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p ${conda_home} \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install conda packages.
ENV PATH="${conda_home}/bin:${PATH}"
RUN conda update -qy conda \
    && conda install -ky -c conda-forge -c bioconda \
      openmm=7.5.1 \
      pytorch=1.9.1 cudatoolkit=${CUDA_VERSION} \
      pdbfixer \
      pip \
      python=3.7


ENV profold_home=/app/profold2
COPY . ${profold_home}
# RUN wget -q -P ${profold_home}/profold2/common/ \
#   https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Install pip packages.
RUN pip3 install --upgrade pip \
    && pip3 install -r ${profold_home}/requirements.txt

# Apply OpenMM patch.
WORKDIR ${conda_home}/lib/python3.7/site-packages
RUN patch -p0 < ${profold_home}/docker/openmm.patch

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.

WORKDIR ${profold_home}
RUN echo -e "#!/bin/bash\n\
cd ${profold_home}\n\
ldconfig\n\
python predictor.py \$@"> /app/run_profold.sh \
    && chmod +x /app/run_profold.sh
ENTRYPOINT ["/app/run_profold.sh"]
