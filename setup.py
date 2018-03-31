import os
import re
from pip.req import parse_requirements
from setuptools import setup, find_packages

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='hack')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


def read_version():
    with open('pytorch_es/__init__.py') as f:
        return re.search(r'__version__ = \'(.+)\'$', f.readline()).group(1)


setup(
    name='pytorch_es',
    version=read_version(),
    license='MIT',
    description='Evolutionary Strategies using PyTorch',
    long_description=read('README.md'),
    url='https://github.com/staturecrane/PyTorch-ES',
    author='Richard Herbert',
    author_email='richard.alan.herbert@gmail.com',
    packages=find_packages(),
    install_requires=reqs,
    keywords=["machine learning", "ai", "evolutionary strategies", "reinforcement learning", "pytorch"],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Internet :: WWW/HTTP',
    ],
)