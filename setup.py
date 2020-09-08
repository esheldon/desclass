from setuptools import setup, find_packages
from glob import glob

scripts = glob('bin/*')
scripts = [s for s in scripts if '~' not in s]

setup(
    name="desclass",
    version="0.1.0",
    packages=find_packages(),
    scripts=scripts,
    author='Erin Sheldon',
    author_email='erin.sheldon@gmail.com',
    url='https://github.com/esheldon/desclass',
)
