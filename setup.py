from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("hello", ["hello.pyx"])]

setup(
  name = 'Hello world app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = [Extension(
      "bg",
      ["bg.pyx"],
      include_dirs=[np.get_include()])]
  )
