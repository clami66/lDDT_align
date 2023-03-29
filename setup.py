from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[
        Extension(
            "lDDT_align.dynamic_programming", sources=["src/lDDT_align/dynamic_programming.pyx"],
            include_dirs = [numpy.get_include()],
        ),
    ],
)
