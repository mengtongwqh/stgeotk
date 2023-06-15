import setuptools
from setuptools import setup

PROJECT = "stgeotk"
DESCRIPTION = "Structural Geology Toolkit"
LONG_DESCRIPTION = "Structural geology algorithms for data analysis and visualization"
DEPENDENCIES = [
        'numpy',
        'matplotlib',
        'scipy']


setup(
  name='stgeotk',
  version='0.1.0',
  author='WU,Qihang',
  author_email='wu.qihang@hotmail.com',
  packages = ['stgeotk'],
  install_requires = DEPENDENCIES,
  description = DESCRIPTION,
  long_description = LONG_DESCRIPTION)
