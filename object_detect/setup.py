from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'object_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Install package.xml and other metadata
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='conner',
    maintainer_email='97805427+GoldenApple265@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    entry_points={
    'console_scripts': [
        'exclusion_zones = object_detect.exclusion_zones:main',
    ],
    }
)
