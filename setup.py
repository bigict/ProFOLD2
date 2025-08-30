"""Install script for setuptools."""

from setuptools import setup, find_packages

from profold2.model import kernel


setup(
    name='ProFOLD2',
    packages=find_packages(),
    version='0.1.0',
    license='MIT',
    description='ProFOLD2 - A protein 3D structure prediction application',
    author='Chungong Yu, Dongbo Bu',
    author_email='chungongyu@gmail.com, dbu@gmail.com',
    url='https://github.com/bigict/ProFOLD2',
    keywords=[
        'artificial intelligence', 'attention mechanism', 'protein folding'
    ],
    install_requires=[
        'biopython',
        'einops>=0.3',
        'openmm>=8.0.0',
        'pdbfixer',
        'tensorboard',
        'torch>=2.3.1',
        'tqdm'
    ],
    test_suite='tests',
    include_package_data=True,
    package_data={
        'profold2': ['model/kernel/csrc/*'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
    **kernel.setuptools(),
    entry_points={
        'console_scripts': [
            'profold2 = profold2.command.main:main'
        ]
    },
)
