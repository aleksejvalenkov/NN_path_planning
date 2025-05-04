from pathlib import Path
from setuptools import setup, find_packages

description = ['NN_path_planning']

# Setting file paths
root = Path(__file__).parent
readme = root / 'README.md'
reqs_main = root / 'requirements.txt'


reqs = [reqs_main]

with open(readme, 'r', encoding='utf-8') as f:
    readme = f.read()
    
dependencies = []    
for req in reqs:
    with open(req, 'r') as f:
        dependencies.extend(f.read().split('\n'))


setup(
    name='NN_path_planning',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=dependencies,
    author='aleksejvalenkov',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/aleksejvalenkov/NN_path_planning',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
