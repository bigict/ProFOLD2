"""Install script for setuptools."""

from setuptools import setup, find_packages

setup(
  name = 'ProFOLD2',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'ProFOLD2 - A protein 3D structure prediction application',
  author = 'Chungong Yu, Dongbo Bu',
  author_email = 'chungongyu@gmail.com, dbu@gmail.com',
  url = 'https://github.com/bigict/ProFOLD2',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'protein folding'
  ],
  install_requires=[
    'biopython',
    'einops>=0.3',
    'numpy',
    'torch>=1.6'
  ],
  test_suite = 'tests',
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
