#  IMPC and NMPC for quadrotor PX4-ROS2-GAZEBO Simulation
This project includes two nodes. The node 'nmpc_output_thrust_torque' is the quadrotor NMPC control simulation node, while 'nnc_quad' is the IMPC control simulation node.
The control variables of this program are the total thrust of the quadrotor and the three-axis control moments, utilizing the OffBoard mode for the total thrust and three-axis control moments of PX4.
This project partially utilizes code from project https://github.com/evannsmc/nmpc_acados_px4.git.


## Prerequisites

Install the following:
- **pyJoules** 
- **ACADOS** 
- **ACADOS Python interface** (`acados_template`)
- **Micro XRCE-DDS Agent**
- **px4_msgs**
- **PX4-Autopilot**
- **PyTorch**
- **Numpy**
- **...**(Other commonly used packages)

The dependencies are all commonly used packages for PX4-ROS2-GAZEBO simulation. 
For the installation of ACADOS, please refer to project https://github.com/evannsmc/nmpc_acados_px4.git. 

## Clone Project
1. Clone directory (**NMPC_PX4_PKG_NN**) into your ROS2 workspace's source directory and build:
```bash
cd <your_ros2_ws/src>
git init
git clone git@github.com/xxx.git
cd ..
colcon build --symlink-install
```

## Modify the parameters of the x500 model
```bash
cp your_ros2_ws/src/NMPC_PX4_PKG_NN/nmpc_px4_pkg/X500_config/model_x500.sdf /<your_Installation location>/PX4-Autopilot/Tools/simulation/gz/models/x500/
cd /<your_Installation location>/PX4-Autopilot/Tools/simulation/gz/models/x500
mv model_x500.sdf model.sdf

cp your_ros2_ws/src/NMPC_PX4_PKG_NN/nmpc_px4_pkg/X500_config/model_x500base.sdf /<your_Installation location>/PX4-Autopilot/Tools/simulation/gz/models/x500_base/
cd /<your_Installation location>/PX4-Autopilot/Tools/simulation/gz/models/x500_base
mv model_x500base.sdf model.sdf
```

## How to  run:
```bash
1. MicroXRCEAgent udp4 --port 8888
2. cd /<your_Installation location>/PX4-Autopilot
   export PX4_GZ_WORLD=lawn
   make px4_sitl gz_x500
run NMPC sim:
3. ros2 run nmpc_px4_pkg nmpc_output_thrust_torque log.log
run IMPC sim:
4. ros2 run nmpc_px4_pkg nnc_quad log.log
5. run plotTest.py to plot results
```

Other usage methods can be found in the code comments.









