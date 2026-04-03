from setuptools import find_packages, setup

package_name = 'realsense_depth'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'pyrealsense2'
    ],
    zip_safe=True,
    maintainer='deb',
    maintainer_email='dblaufus@terpmail.umd.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'realsense_depth = realsense_depth.realsense_camera:main'
        ],
    },
)
