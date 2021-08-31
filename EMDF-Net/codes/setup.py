"""Setup module for Iris."""

from setuptools import setup, find_packages

setup(
        name='ScriptPrediction',
        version='0.1',
        description='Script Prediction using EMDF-Net.',

        author='Pengpeng Zhou, Caiyong Wang',
        author_email='zhoupengpeng@bupt.edu.cn, wangcaiyong@bucea.edu.cn',

        packages=find_packages(exclude=[]),
        python_requires='>=3.6',
        install_requires=[
            'coloredlogs',
            'numpy',
            'torch>=1.8.0',
            'scipy',
        ],
)
