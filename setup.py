from setuptools import setup, find_packages

setup(
    name='diffusion_modeller',
    version='0.1.0',
    author='Andrew Gorin',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='A package for modelling diffusion'
)