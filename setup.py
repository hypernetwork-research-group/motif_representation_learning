from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = "gcc"
os.environ["CXX"] = "clang++"

extensions = [
    Extension("__motif__/*", ["__motif__/*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3', '-march=native', '-ffast-math', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]
setup(
    name="Mochy",
    ext_modules=cythonize(extensions),
)
