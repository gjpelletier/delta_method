from setuptools import setup
import sys
sys.path.insert(0, ".")
from delta_method import __version__

setup(
    name='delta_method',
    version=__version__,
    author='Greg Pelletier',
    py_modules=['delta_method'], 
    install_requires=['numpy','scipy'],
)