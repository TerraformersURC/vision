from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'image_handler'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config', 'calibration_images'), glob('config/calibration_images/*')),
        (os.path.join('share', package_name, 'config'), ['config/camera_calibration_parameters.npz']),
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
        'image_overlay = image_handler.image_overlay:main',
        'calibrate = image_handler.calibrate:main',
        'camera_manager = image_handler.camera_manager:main',
        'create_panorama = image_handler.create_panorama:main'
    ],
    }
)
