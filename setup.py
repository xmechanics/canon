from setuptools import setup

import canon

setup(name='Canon',
      version=canon.__verison__,
      description='A python package for X-Ray Laue diffractometer',
      url='http://github.com/structrans/Canon',
      author=('Xian Chen', 'Yintao Song'),
      author_email=('xianchen@ust.hk', 'yintaosong@gmail.com'),
      license='LICENSE',
      classifiers = (
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: Freeware',
          'Natural Language :: English',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: Implementation :: CPython'
      ),
      packages=(
          'canon',
      ),
      scripts = [],
      install_requires=(
          'numpy >= 1.9.0',
          'skimage >= 0.11.3'
      ),
      test_suite='nose.collector',
      tests_require = ['nose']
      )
