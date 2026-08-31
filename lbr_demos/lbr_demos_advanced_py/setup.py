import glob

from setuptools import setup

package_name = "lbr_demos_advanced_py"

setup(
    name=package_name,
    version="2.2.1",
    packages=[package_name],
    package_data={package_name: ["*.pkl", "*.jpg"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mhubii",
    maintainer_email="m.huber_1994@hotmail.de",
    description="Advanced Python demos for the lbr_ros2_control.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "admittance_control = lbr_demos_advanced_py.admittance_control_node:main",
            "admittance_rcm_control = lbr_demos_advanced_py.admittance_rcm_control_node:main",
            "pose_planning_2 = lbr_demos_advanced_py.pose_planning_node_2:main",
            "pose_planning_3 = lbr_demos_advanced_py.pose_planning_node_3:main",
            "pose_planning_4 = lbr_demos_advanced_py.pose_planning_node_4:main",
            "pose_planning_5 = lbr_demos_advanced_py.pose_planning_node_5:main",
            "pose_planning_6 = lbr_demos_advanced_py.pose_planning_node_6:main",
            "state_perception = lbr_demos_advanced_py.state_perception:main",
            "pose_planning_7 = lbr_demos_advanced_py.pose_planning_node_7:main",
            "pose_planning_8 = lbr_demos_advanced_py.pose_planning_node_8:main",
            "pose_planning_9 = lbr_demos_advanced_py.pose_planning_node_9:main",
            "pose_planning_10 = lbr_demos_advanced_py.pose_planning_node_10:main",
            "pose_planning_node_realtime_contact_line = lbr_demos_advanced_py.pose_planning_node_realtime_contact_line:main",
            "pose_planning_contrast = lbr_demos_advanced_py.pose_planning_node_contrast:main",
            "contact_posint_estimate = lbr_demos_advanced_py.contact_pose_estimate:main",
            "pose_planning_mapping = lbr_demos_advanced_py.pose_planning_node_mapping:main",
            "pose_planning_node_heuristic_search = lbr_demos_advanced_py.pose_planning_node_heuristic_search:main",
            "pose_planning_node_heuristic_search_push = lbr_demos_advanced_py.pose_planning_node_heuristic_search_push:main",
        ],
    },
)
