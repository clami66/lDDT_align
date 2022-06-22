from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "lDDT_align.dynamic_programming", sources=["src/lDDT_align/dynamic_programming.pyx"],
        ),
    ],
)
