import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

import canon


# noinspection PyAttributeOutsideInit
class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(name='Canon',
      version=canon.__verison__,
      description='A python package for X-Ray Laue diffractometer',
      url='http://github.com/structrans/Canon',
      author=('Xian Chen', 'Yintao Song'),
      author_email=('xianchen@ust.hk', 'yintaosong@gmail.com'),
      license='LICENSE',
      classifiers=(
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: Freeware',
          'Natural Language :: English',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: Implementation :: CPython'
      ),
      packages=('canon', 'canon.dat', 'canon.img', 'canon.mpi', 'canon.pattern', 'canon.seq', 'canon.util'),
      scripts=[],
      install_requires=(
          'numpy',
          'scipy',
          'scikit-image'),
      tests_require=['pytest'],
      cmdclass={'test': PyTest}
      )
