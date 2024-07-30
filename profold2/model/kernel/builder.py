import os
import importlib
import subprocess
import logging

import torch
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension, load

from profold2.utils import package_dir


logger = logging.getLogger(__name__)


def installed_cuda_version():
  assert CUDA_HOME and torch.version.cuda
  raw_output = subprocess.check_output([CUDA_HOME + '/bin/nvcc', '-V'],
                                       universal_newlines=True)
  output = raw_output.split()
  release_idx = output.index('release') + 1
  release = output[release_idx].split('.')
  major_version = release[0]
  minor_version = release[1][0]

  return int(major_version), int(minor_version)


version_dependent_macros = [
    '-DVERSION_GE_1_1',
    '-DVERSION_GE_1_3',
    '-DVERSION_GE_1_5',
]


def extra_compile_args(jit_mode=True):
  cxx_args = ['-O3', '-std=c++17'] + version_dependent_macros

  nvcc_threads = min(os.cpu_count(), 8)
  if 'PF_NVCC_THREADS' in os.environ:
    nvcc_threads = int(os.environ['PF_NVCC_THREADS'])

  nvcc_args = [
      '-O3', '--use_fast_math', '-std=c++17', f'--threads={nvcc_threads}',
      '-U__CUDA_NO_HALF_OPERATORS__',
      '-U__CUDA_NO_HALF_CONVERSIONS__',
      '-U__CUDA_NO_HALF2_OPERATORS__',
  ]

  nvcc_args += version_dependent_macros

  compute_capabilities = []
  if jit_mode:
    # Compile for underlying architectures since we know those at runtime
    for i in range(torch.cuda.device_count()):
      cc_major, cc_minor = torch.cuda.get_device_capability(i)
      cc = f'{cc_major}.{cc_minor}'
      if cc not in compute_capabilities:
        compute_capabilities.append(cc)
    compute_capabilities = sorted(compute_capabilities)
    if compute_capabilities:
      compute_capabilities[-1] += '+PTX'
  else:
    # Cross-compile mode, compile for various architectures
    # env override takes priority
    cross_compile_archs = '6.0;6.1;7.0'
    cuda_major, _ = installed_cuda_version()
    if cuda_major >= 11:
      cross_compile_archs = f'{cross_compile_archs};8.0'
    cross_compile_archs_env = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if cross_compile_archs_env is not None:
      logger.warning(
          'env var `TORCH_CUDA_ARCH_LIST=%s` overrides `cross_compile_archs=%s`',  # pylint: disable=line-too-long
          cross_compile_archs_env, cross_compile_archs)
      cross_compile_archs = ';'.join(cross_compile_archs_env.split())
    compute_capabilities = cross_compile_archs.split(';')

  enable_bf16 = True
  for cc in compute_capabilities:
    major, minor = cc.split('.')
    if minor.endswith('+PTX'):
      minor = minor[:-4]
      nvcc_args.append(
          f'-gencode=arch=compute_{major}{minor},code=compute_{major}{minor}')
    nvcc_args.extend([
        '-gencode',
        f'arch=compute_{major}{minor},code=sm_{major}{minor}',
    ])
    if int(major) <= 7:
      enable_bf16 = False
  if enable_bf16:
    cxx_args.append('-DBF16_AVAILABLE')
    nvcc_args.append('-DBF16_AVAILABLE')

  major = torch.cuda.get_device_properties(0).major  #ignore-cuda
  minor = torch.cuda.get_device_properties(0).minor  #ignore-cuda
  nvcc_args.append(f'-DGPU_ARCH={major}{minor}')

  return cxx_args, nvcc_args


ATTENTION_CORE_NAME = 'evoformer_attn'
ATTENTION_CORE_SRC = [
    os.path.join('csrc', 'attention.cpp'),
    os.path.join('csrc', 'attention_back.cu'),
    os.path.join('csrc', 'attention_cu.cu'),
]

if 'CUTLASS_PATH' not in os.environ:
  def_cutlass_path = os.path.join(package_dir(), 'cutlass')
  logger.warning(
      'You can specify the environment variable $CUTLASS_PATH to override `%s`',
      def_cutlass_path)
  os.environ['CUTLASS_PATH'] = def_cutlass_path

ATTENTION_CORE_INC = [
    os.path.join(os.environ['CUTLASS_PATH'], 'include'),
    os.path.join(os.environ['CUTLASS_PATH'], 'tools/util/include'),
]

_loaded_ops = {}


def is_compatible():
  if CUDA_HOME and torch.version.cuda:
    cuda_okay = True
    if torch.cuda.is_available():
      sys_cuda_major, _ = installed_cuda_version()
      torch_cuda_major = int(torch.version.cuda.split('.')[0])  # pylint: disable=use-maxsplit-arg
      cuda_capability = torch.cuda.get_device_properties(0).major
      if cuda_capability < 7:
        logger.warning('Please use a GPU with compute capability >= 7.0')
        cuda_okay = False
      if torch_cuda_major < 11 or sys_cuda_major < 11:
        logger.warning('Please use CUDA 11+')
        cuda_okay = False
    return cuda_okay
  return False


def build(name, **kwargs):
  assert is_compatible()

  if name in _loaded_ops:
    return _loaded_ops[name]

  # Ensure the op we're about to load was compiled with the same
  # torch/cuda versions we are currently using at runtime.
  try:
    op_module = importlib.import_module(name)
    _loaded_ops[name] = op_module
    return op_module
  except ModuleNotFoundError:
    pass

  if name == ATTENTION_CORE_NAME:
    sources, includes = ATTENTION_CORE_SRC, ATTENTION_CORE_INC
  else:
    assert False, name

  pwd = os.path.dirname(__file__)
  sources = [os.path.join(pwd, src) for src in sources]

  verbose = True
  if 'verbose' in kwargs:
    verbose = kwargs.pop('verbose')

  cxx_args, nvcc_args = extra_compile_args(jit_mode=True)
  op_module = load(name=name,
                   sources=sources,
                   extra_include_paths=includes,
                   extra_cflags=cxx_args,
                   extra_cuda_cflags=nvcc_args,
                   verbose=verbose,
                   **kwargs)
  _loaded_ops[name] = op_module
  return op_module


def setuptools(**kwargs):
  if is_compatible():
    cxx_args, nvcc_args = extra_compile_args(jit_mode=False)
    sources, includes = ATTENTION_CORE_SRC, ATTENTION_CORE_INC

    pwd = os.path.join(*__package__.split('.'))
    sources = [os.path.join(pwd, src) for src in sources]

    modules = [
        CUDAExtension(name=ATTENTION_CORE_NAME,
                      sources=sources,
                      include_dirs=includes,
                      extra_compile_args={
                          'cxx': cxx_args,
                          'nvcc': nvcc_args
                      },
                      **kwargs)
    ]
    return {'ext_modules': modules, 'cmdclass': {'build_ext': BuildExtension}}
  return {}
