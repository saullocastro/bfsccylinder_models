from __future__ import absolute_import

import os
import inspect
import subprocess
from setuptools import setup, find_packages
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize


is_released = True
version = '0.2.34'


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info(version, is_released):
    fullversion = version
    if not is_released:
        git_revision = git_version()
        fullversion += '.dev0+' + git_revision[:7]
    return fullversion


def write_version_py(version, is_released, filename='bfsccylinder_models/version.py'):
    fullversion = get_version_info(version, is_released)
    with open("./bfsccylinder_models/version.py", "wb") as f:
        f.write(b'__version__ = "%s"\n' % fullversion.encode())
    return fullversion


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    setupdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return open(os.path.join(setupdir, fname)).read()


#_____________________________________________________________________________

install_requires = [
        "bfsccylinder",
        "numpy",
        "scipy",
        "composites",
        ]

#Trove classifiers
CLASSIFIERS = """\

Development Status :: 4 - Beta
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
Topic :: Scientific/Engineering :: Mathematics
Topic :: Education
License :: OSI Approved :: BSD License
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Operating System :: Microsoft :: Windows
Operating System :: Unix

"""

fullversion = write_version_py(version, is_released)

if os.name == 'nt':
    compile_args = ['/openmp', '/O2']
    link_args = []
else:
    compile_args = ['-fopenmp', '-static', '-static-libgcc', '-static-libstdc++']
    link_args = ['-fopenmp', '-static-libgcc', '-static-libstdc++']

if 'CYTHON_TRACE_NOGIL' in os.environ.keys():
    compile_args = ['-O0']
    link_args = []

include_dirs = [
            np.get_include(),
            ]

extensions = [
    Extension('bfsccylinder_models.linbuck_VAFW',
        sources=[
            './bfsccylinder_models/linbuck_VAFW.pyx',
            ],
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++'),
    Extension('bfsccylinder_models.koiter_cylinder_CTS',
        sources=[
            './bfsccylinder_models/koiter_cylinder_CTS.pyx',
            ],
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++'),
    Extension('bfsccylinder_models.koiter_cylinder',
        sources=[
            './bfsccylinder_models/koiter_cylinder.pyx',
            ],
        include_dirs=include_dirs,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language='c++'),
    ]

ext_modules = cythonize(extensions,
        compiler_directives={'linetrace': True},
        language_level = '3',
        )

data_files = [('', [
        'README.md',
        'LICENSE',
        ])]

s = setup(
    name = "bfsccylinder_models",
    version = fullversion,
    author = "Saullo G. P. Castro",
    author_email = "S.G.P.Castro@tudelft.nl",
    description = ("Models using the BFSCCylinder finite element"),
    license = "3-Clause BSD",
    keywords = "finite elements shell structural analysis buckling vibration dynamics",
    url = "https://github.com/saullocastro/bfsccylinder_models",
    data_files=data_files,
    long_description=read('README.md'),
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires=install_requires,
    ext_modules = ext_modules,
    include_package_data=True,
    packages=find_packages(),
)

