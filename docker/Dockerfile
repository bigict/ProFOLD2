ARG CUDA=12.1.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu22.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
      hmmer \
      kalign \
      tzdata \
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
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS="yes"
RUN conda install -ky pip python=3.11


ENV profold_home=/app/profold2
COPY . ${profold_home}

# Install dependencies
ENV cuda_version=${CUDA}
RUN bash ${profold_home}/install_env.sh

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
${profold_home}/examples/predict.job \$@"> /app/run_profold.sh \
    && chmod +x /app/run_profold.sh
ENTRYPOINT ["/app/run_profold.sh"]
