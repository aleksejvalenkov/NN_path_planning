from pathlib import Path
from setuptools import setup, find_packages

description = ['pattern_based_calibration']

# Setting file paths
root = Path(__file__).parent
readme = root / 'README.md'
reqs_main = root / 'requirements.txt'
reqs_calibration_tools = root / 'packages/calibration_tools/requirements.txt'
reqs_realsense_tools = root / 'packages/realsense_tools/requirements.txt'
reqs_zed_tools = root / 'packages/zed_tools/requirements.txt'
reqs_azure_tools = root / 'packages/azure_tools/requirements.txt'

reqs = [reqs_main,reqs_calibration_tools, reqs_realsense_tools, reqs_zed_tools, reqs_azure_tools]

with open(readme, 'r', encoding='utf-8') as f:
    readme = f.read()
    
dependencies = []    
for req in reqs:
    with open(req, 'r') as f:
        dependencies.extend(f.read().split('\n'))


setup(
    name='pattern_based_calibration',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=dependencies,
    author='Aleksei_Valenkov',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://git.sberrobots.ru/vedu/doedu/-/tree/calib_3d_sensors/calibration/pattern_based_calibration?ref_type=heads',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
