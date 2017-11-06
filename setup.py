#!/usr/bin/env python3
from glob import glob
from distutils.core import setup

setup(name='comadyn',
      version='0.1',
      description='Cometary coma dynamics integrator.',
      author='Michael S. P. Kelley',
      author_email='msk@astro.umd.edu',
      url='https://github.com/mkelley/comadyn',
      packages=['comadyn'],
      requires=['numpy', 'mskpy'],
      license='BSD',
      classifiers = [
              'Intended Audience :: Science/Research',
              "License :: OSI Approved :: BSD License",
              'Operating System :: OS Independent',
              "Programming Language :: Python :: 3",
              'Topic :: Scientific/Engineering :: Astronomy'
          ]
     )
