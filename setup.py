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
    version='0.1.0',
    author='Gabriel Synnaeve',
    packages=['abnet', 'abnet.utils'],
    modules=['abnet.prepare', 'abnet.utils.stack_fbanks',
             'abnet.dataset_iterators', 'abnet.layers', 'abnet.sampler',
             'abnet.nnet_archs', 'abnet.classifiers', 'abnet.run',
             'abnet.utils.do_fbank'],
    # url='http://pypi.python.org/pypi/ABnet/',
    license='license/LICENSE.txt',
    description='ABNet is a "same/different"-based loss trained neural net',
    long_description=open('README.rst').read(),
    install_requires=[
        "python >= 2.7",
        "numpy >= 1.8.0",
        "scipy >= 0.13.0",
        "theano >= 0.6.0",
        "joblib >= 0.8.4",
    ],
    # cmdclass = {'test': PyTest},
)
