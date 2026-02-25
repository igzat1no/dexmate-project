from setuptools import find_packages, setup
import os
from glob import glob

package_name = "dexmate_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.rviz")),
        (os.path.join("lib", package_name, "workers"), glob("scripts/workers/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Zongtai Li",
    maintainer_email="zongtai@example.com",
    description="ROS 2 nodes for Dexmate pick-and-place system",
    license="MIT",
    scripts=[
        "scripts/scene_node",
        "scripts/planner_node",
        "scripts/demo_client",
        "scripts/interactive_viz",
    ],
)
