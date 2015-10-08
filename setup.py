try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.core import Command
from distutils.extension import Extension
from Cython.Build import cythonize
import os
import numpy


class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys,subprocess
        try:
            import pytest
        except ImportError:
            raise(ImportError, 'You need to have pytest to run the tests,'
                  ' try installing it (pip install pytest)')
        errno = subprocess.call(['py.test', '-s', 'ABXpy/test'])
        raise SystemExit(errno)


setup(
    name='abnet',
    version='interspeech2015',
    author='Gabriel Synnaeve',
    packages=['abnet', 'abnet.utils'],
    modules=['abnet.dataset_iterators', 'abnet.classifiers',
             'abnet.layers', 'abnet.nnet_archs', 'abnet.prepare',
             'abnet.train', 'abnet.utils.do_fbanks',
             'abnet.utils.stack_fbanks'],
    # url='http://pypi.python.org/pypi/ABnet/',
    license='license/LICENSE.txt',
    description='ABNet is a "same/different"-based loss trained neural net',
    long_description=open('README.md').read(),
    dependency_links = [
        'http://github.com/bootphon/spectral/tarball/master/#egg=mwv-spectral',
        'https://github.com/SnippyHolloW/DTW_Cython/tarball/master/#egg=snippy-dtw'],
    install_requires=[
        "python >= 2.7",
        "numpy >= 1.8.0",
        "scipy >= 0.13.0",
        "theano >= 0.6.0",
        "joblib >= 0.8.4",
        "mwv-spectral",
        "snippy-dtw",
    ],
    # cmdclass = {'test': PyTest},
)
