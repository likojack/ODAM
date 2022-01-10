"""Setup learnable_primitives"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from itertools import dropwhile
import numpy as np
from os import path


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("learnable_primitives", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta


def get_extensions():
    return cythonize([
        Extension(
            "learnable_primitives.fast_sampler._sampler",
            [
                "learnable_primitives/fast_sampler/_sampler.pyx",
                "learnable_primitives/fast_sampler/sampling.cpp"
            ],
            language="c++11",
            libraries=["stdc++"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-std=c++11", "-O3"]
        )
    ])

def setup_package():
    meta = collect_metadata()
    setup(
        name="learnable_primitives",
        version=meta["version"],
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
        ],
        ext_modules=get_extensions()
    )


if __name__ == "__main__":
    setup_package()
