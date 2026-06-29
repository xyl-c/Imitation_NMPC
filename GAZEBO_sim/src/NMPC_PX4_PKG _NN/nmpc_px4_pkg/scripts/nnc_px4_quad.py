
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand, VehicleStatus, VehicleOdometry, TrajectorySetpoint, RcChannels
from std_msgs.msg import Float64MultiArray
from px4_msgs.msg import ActuatorMotors, VehicleThrustSetpoint, VehicleTorqueSetpoint

import os
import sys
import traceback

from transforms3d.euler import quat2euler
import math as m
import numpy as np
import time

from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain
from pyJoules.energy_meter import EnergyContext

from nmpc_px4_pkg.Logger_pqr import Logger

import torch
from ament_index_python.packages import get_package_share_directory
import mpc_BC_network_3

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""
    def __init__(self) -> None:
        super().__init__('offboard_control_takeoff_and_land')
        self.mocap_k = -1
        self.full_rotations = 0
        self.made_it = 0
###############################################################################################################################################

        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        self.sim = bool(int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: ")))
        print(f"{'SIMULATION' if self.sim else 'HARDWARE'}")
        self.select_tra = int(input("which Trajectories?  1: Circle 2: fig-8 3: helix 4: triangle :"))
        self.double_speed = bool(int(input("Double Speed Trajectories? Press 1 for Yes and 0 for No: ")))
        self.select_ctrl = bool(int(input("Pytorch or Numpy? Press 0 for Pytorch and 1 for Numpy: ")))

        self.ctrl_loop_time_log = []
        self.x_log, self.y_log, self.z_log, self.yaw_log = [], [], [], []
        self.throttle_log, self.roll_log, self.pitch_log = [], [], []
        self.p_u_log, self.q_u_log, self.r_u_log = [], [], []
        self.x_ref_log, self.y_ref_log, self.z_ref_log, self.yaw_ref_log = [], [], [], []
        self.vx_log, self.vy_log, self.vz_log, self.p_log, self.q_log, self.r_log = [], [], [], [], [], []
        self.mpc_timel_array_log = []
        self.ctrl_callback_timel_log = []
        self.final = []

        self.mode_channel = 5
        self.pyjoules_on = False # int(input("Use PyJoules? 1 for Yes 0 for No: ")) #False
        if self.pyjoules_on:
            self.csv_handler = CSVHandler('mpc_cpu_energy.csv')
###############################################################################################################################################

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


        # Create Publishers
        # Publishers for Setting to Offboard Mode and Arming/Diasarming/Landing/etc
        self.offboard_control_mode_publisher = self.create_publisher( #publishes offboard control heartbeat
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher( #publishes vehicle commands (arm, offboard, disarm, etc)
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        # Publishers for Sending Setpoints in Offboard Mode: 1) Body Rates and Thrust, 2) Position and Yaw 
        self.actuator_thrust_publisher = self.create_publisher( #publishes body rates and thrust setpoint
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.actuator_torque_publisher = self.create_publisher( #publishes body rates and torque setpoint
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        
        self.trajectory_setpoint_publisher = self.create_publisher( #publishes trajectory setpoint
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
        # Publisher for Logging States, Inputs, and Reference Trajectories for Data Analysis
        self.state_input_ref_log_publisher_ = self.create_publisher( #publishes log of states and input
            Float64MultiArray, '/state_input_ref_log', 10)
        self.state_input_ref_log_msg = Float64MultiArray() #creates message for log of states and input        

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription( #subscribes to vehicle status (arm, offboard, disarm, etc)
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)
    
        self.offboard_mode_rc_switch_on = True if self.sim else False #Offboard mode starts on if in Sim, turn off and wait for RC if in hardware
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" to make sure we'd like position vs offboard vs land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_callback, qos_profile
        )

###############################################################################################################################################

        # Initialize variables:  
        self.cushion_time = 5.0
        self.flight_time = 20.0
        self.time_before_land = self.flight_time + 2.0*(self.cushion_time)
        print(f"time_before_land: {self.time_before_land}")
        self.offboard_setpoint_counter = 0 #helps us count 10 cycles of sending offboard heartbeat before switching to offboard mode and arming
        self.vehicle_status = VehicleStatus() #vehicle status variable to make sure we're in offboard mode before sending setpoints

        self.T0 = time.time() # initial time of program
        self.time_from_start = time.time() - self.T0 # time from start of program initialized and updated later to keep track of current time in program
        self.first_iteration = True #boolean to help us initialize the first iteration of the program

        if self.select_tra == 1:
            self.ref_fun = self.circle_horz_ref_func
        elif self.select_tra == 2:
            self.ref_fun = self.fig8_horz_ref_func
        elif self.select_tra == 3:
            self.ref_fun = self.helix
        elif self.select_tra == 4:
            self.ref_fun = self.triangle_traj
        else:
            print('no such trajectory!!!')
            exit(0)
        ## You can define any other trajectory, 
        ## but in IMPC, the initial position tracking error should be less than [2,2,2]^3. 

###############################################################################################################################################
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.select_ctrl == 0: 
            self.d_mean = torch.tensor([-1.2064e-08,  4.0600e-09, -7.7257e-08, -7.3349e-10, -1.4359e-09,
                    -5.1692e-09, -5.7632e-09,  1.9008e-08, -4.5491e-08,  7.6217e-09,
                    -1.0114e-09,  6.1788e-09,  4.4364e-10], device=self.device)
            self.d_std = torch.tensor([0.3364, 0.3364, 0.3364, 0.7793, 0.7793, 0.8720, 0.8720, 0.9315, 0.9315,
                    0.9315, 0.2498, 0.2498, 0.2498], device=self.device)
            self.d_max = torch.tensor([40.0000,  0.0837,  0.0982,  0.1114], device=self.device)
            self.d_min = torch.tensor([-0.0000, -0.0837, -0.0981, -0.1113], device=self.device)

            pkg_share = get_package_share_directory('nmpc_px4_pkg')
            weight_path = os.path.join(pkg_share, 'models', 'mpcBc_net_298.pth')
            self.nn_controller = torch.load(weight_path, weights_only=False).to(self.device)
            self.nn_controller.eval()


        if self.select_ctrl == 1: 
            self.d_mean = np.array([-1.2064e-08,  4.0600e-09, -7.7257e-08, -7.3349e-10, -1.4359e-09,
                    -5.1692e-09, -5.7632e-09,  1.9008e-08, -4.5491e-08,  7.6217e-09,
                    -1.0114e-09,  6.1788e-09,  4.4364e-10]).astype(np.float32)
            self.d_std = np.array([0.3364, 0.3364, 0.3364, 0.7793, 0.7793, 0.8720, 0.8720, 0.9315, 0.9315,
                    0.9315, 0.2498, 0.2498, 0.2498]).astype(np.float32)
            self.d_max = np.array([40.0000,  0.0837,  0.0982,  0.1114]).astype(np.float32)
            self.d_min = np.array([-0.0000, -0.0837, -0.0981, -0.1113]).astype(np.float32)

            pkg_share = get_package_share_directory('nmpc_px4_pkg')
            weight_path = os.path.join(pkg_share, 'models', 'mpcBc_net_298.pth')
            nn_controller = torch.load(weight_path, weights_only=False).to(self.device)
            nn_controller.eval()
            self.W1 = nn_controller.model[0].weight.detach().cpu().numpy().astype(np.float32)
            self.b1 = nn_controller.model[0].bias.detach().cpu().numpy().astype(np.float32)
            self.W2 = nn_controller.model[2].weight.detach().cpu().numpy().astype(np.float32)
            self.b2 = nn_controller.model[2].bias.detach().cpu().numpy().astype(np.float32)
            self.W3 = nn_controller.model[4].weight.detach().cpu().numpy().astype(np.float32)
            self.b3 = nn_controller.model[4].bias.detach().cpu().numpy().astype(np.float32)
            self.W4 = nn_controller.model[6].weight.detach().cpu().numpy().astype(np.float32)
            self.b4 = nn_controller.model[6].bias.detach().cpu().numpy().astype(np.float32)

     
###############################################################################################################################################
        if self.sim:
            print("Using simulator throttle from force conversion function")
            self.MASS = 2.0 #set simulation mass from iris model sdf for linearized model calculations 1.535
            # The following 3 variables are used to convert between force and throttle commands for the iris drone defined in PX4 stack for gazebo simulation
            # self.MOTOR_CONSTANT = 0.00000584 #iris gazebo simulation motor constant
            # self.MOTOR_CONSTANT = 0.00000854858
            # self.MOTOR_CONSTANT = 0.00000514858
            # self.MOTOR_CONSTANT = 0.000007199265381
            self.MOTOR_CONSTANT = 0.0000076
            self.MOTOR_VELOCITY_ARMED = 100.0 #iris gazebo motor velocity when armed
            self.MOTOR_INPUT_SCALING = 1000.0 #iris gazebo simulation motor input scaling
        elif not self.sim:
            print("Using hardware throttle from force conversion function and certain trajectories will not be available")
            self.MASS = 1.5 #alternate: 1.69kg and have grav_mass = 2.10

        self.GRAVITY = 9.8 #gravity
        self.T_lookahead = 0. #lookahead time for prediction and reference tracking in controller

###############################################################################################################################################
        horizon = 3.0 #2, 3
        num_steps = 20 #10, 20
        self.num_steps = num_steps
        self.horizon = horizon


        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                  '2x Speed' if self.double_speed else '1x Speed',
                                  'Horizon:'+str(horizon),
                                  'Num Steps:'+str(num_steps),
                                  'Pyjoules' if self.pyjoules_on else 'No Pyjoules',
                                  ])
###############################################################################################################################################

        #Create Function @ {1/self.offboard_timer_period}Hz (in my case should be 10Hz/0.1 period) to Publish Offboard Control Heartbeat Signal
        self.offboard_timer_period = 0.1
        self.timer = self.create_timer(self.offboard_timer_period, self.offboard_mode_timer_callback)
        # exit(0)

        # Create Function at {1/self.controller_timer_period}Hz (in my case should be 100Hz/0.01 period) to Send Control Input
        self.controller_timer_period = 1.0 / 1000
        self.timer_ctrl = self.create_timer(self.controller_timer_period, self.controller_timer_callback)


    def relu(self, x):
        return np.maximum(x, 0, out=x)

    def forward_numpy(self, x):
        z1 = self.W1 @ x + self.b1
        a1 = self.relu(z1)

        z2 = self.W2 @ a1 + self.b2
        a2 = self.relu(z2)

        z3 = self.W3 @ a2 + self.b3
        a3 = self.relu(z3)

        z4 = self.W4 @ a3 + self.b4
        a4 = 1 / (1 + np.exp(-z4))
        return a4
    # The following 4 functions all call publish_vehicle_command to arm/disarm/land/ and switch to offboard mode
    # The 5th function publishes the vehicle command
    # The 6th function checks if we're in offboard mode
    # The 7th function handles the safety RC control switches for hardware
    def arm(self): #1. Sends arm command to vehicle via publish_vehicle_command function
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self): #2. Sends disarm command to vehicle via publish_vehicle_command function
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self): #3. Sends offboard command to vehicle via publish_vehicle_command function
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self): #4. Sends land command to vehicle via publish_vehicle_command function
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_vehicle_command(self, command, **params) -> None: #5. Called by the above 4 functions to send parameter/mode commands to the vehicle
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def vehicle_status_callback(self, vehicle_status): #6. This function helps us check if we're in offboard mode before we start sending setpoints
        """Callback function for vehicle_status topic subscriber."""
        # print('vehicle status callback')
        self.vehicle_status = vehicle_status

    def rc_channel_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print('rc channel callback')
        self.mode_channel = 5
        flight_mode = rc_channels.channels[self.mode_channel-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on = True if flight_mode >= 0.75 else False

    # The following 2 functions are used to publish offboard control heartbeat signals
    def publish_offboard_control_heartbeat_signal2(self): #1)Offboard Signal2 for Returning to Origin with Position Control
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_offboard_control_heartbeat_signal1(self): #2)Offboard Signal1 for Newton-Rapshon Body Rate Control
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = True
        msg.direct_actuator = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)


# ~~ The remaining functions are all intimately related to the MPC Control Algorithm ~~
    # The following 2 functions are used to convert between force and throttle commands
    def get_throttle_command_from_force(self, collective_thrust): #Converts force to throttle command
        """ Convert the collective thrust to a throttle command. """
        collective_thrust = -collective_thrust
        # print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
        if self.sim:
            motor_speed = np.sqrt(collective_thrust / (4.0 * self.MOTOR_CONSTANT))
            throttle_command = (motor_speed - self.MOTOR_VELOCITY_ARMED) / self.MOTOR_INPUT_SCALING
            return -throttle_command
        
        if not self.sim:
            a = 0.00705385408507030
            b = 0.0807474474438391
            c = 0.0252575818743285

            # equation form is a*x + b*sqrt(x) + c = y
            throttle_command = a*collective_thrust + b*m.sqrt(collective_thrust) + c
            return throttle_command

    def get_force_from_throttle_command(self, throttle_command): #Converts throttle command to force
        """ Convert the throttle command to a collective thrust. """
        throttle_command = -throttle_command
        print(f"Conv2Force: throttle_command: {throttle_command}")
        if self.sim:
            motor_speed = (throttle_command * self.MOTOR_INPUT_SCALING) + self.MOTOR_VELOCITY_ARMED
            collective_thrust = 4.0 * self.MOTOR_CONSTANT * motor_speed ** 2
            return -collective_thrust
        
        if not self.sim:
            a = 19.2463167420814
            b = 41.8467162352942
            c = -7.19353022443441

            # equation form is a*x^2 + b*x + c = y
            collective_thrust = a*throttle_command**2 + b*throttle_command + c
            return -collective_thrust


    def normalize_angle(self, angle):
        """ Normalize the angle to the range [-pi, pi]. """
        return m.atan2(m.sin(angle), m.cos(angle))


    def adjust_yaw(self, yaw):
        mocap_psi = yaw
        self.mocap_k += 1
        psi = None
        
        if self.mocap_k == 0:
            self.prev_mocap_psi = mocap_psi
            psi = mocap_psi

        elif self.mocap_k > 0:
            # mocap angles are from -pi to pi, whereas the angle state variable in the MPC is an absolute angle (i.e. no modulus)
            # I correct for this discrepancy here
            if self.prev_mocap_psi > np.pi*0.9 and mocap_psi < -np.pi*0.9:
                # Crossed 180 deg, CCW
                self.full_rotations += 1
            elif self.prev_mocap_psi < -np.pi*0.9 and mocap_psi > np.pi*0.9:
                # Crossed 180 deg, CW
                self.full_rotations -= 1

            psi = mocap_psi + 2*np.pi * self.full_rotations
            self.prev_mocap_psi = mocap_psi
        
        return psi


    def vehicle_odometry_callback(self, msg): # Odometry Callback Function Yields Position, Velocity, and Attitude Data
        """Callback function for vehicle_odometry topic subscriber."""
        # print('vehicle odometry callback')

        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2]

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

        # self.roll, self.pitch, yaw = self.euler_from_quaternion(msg.q)
        self.roll, self.pitch, self.yaw = quat2euler(msg.q)

        # self.yaw = self.adjust_yaw(self.yaw)

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.state_vector = np.array([[self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.vx, self.vy, self.vz, self.p, self.q, self.r]]).T 
        self.nr_state = np.array([[self.x, self.y, self.z, self.yaw]]).T
        # print(f"State Vector: {self.state_vector}")
        # print(f"NR State: {self.nr_state}")

    def publish_thrust_torque_setpoint(self, T: float, Mx: float, My: float, Mz: float): #Publishes Body Rate and Thrust Setpoints
        """Publish the thrust_torque setpoint."""
        thrust_msg = VehicleThrustSetpoint()
        thrust_msg.xyz = [0.0, 0.0, float(T)]

        torque_msg = VehicleTorqueSetpoint()
        torque_msg.xyz = [float(Mx), float(My), float(Mz)]

        thrust_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        torque_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.actuator_thrust_publisher.publish(thrust_msg)
        self.actuator_torque_publisher.publish(torque_msg)

        # print("in publish rates setpoint")
        # self.get_logger().info(f"Publishing rates setpoints [r,p,y]: {[roll, pitch, yaw]}")
        print(f"Publishing thrust_torque setpoints [T, Mx, My, Mz]: {[float(T), float(Mx), float(My), float(Mz)]}")

    def publish_position_setpoint(self, x: float, y: float, z: float): #Publishes Position and Yaw Setpoints
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 0.0  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}")


# ~~ The following 2 functions are the main functions that run at 10Hz and 100Hz ~~
    def offboard_mode_timer_callback(self) -> None: # ~~Runs at 10Hz and Sets Vehicle to Offboard Mode  ~~
        """Offboard Callback Function for The 10Hz Timer."""
        # print("In offboard timer callback")

        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            if self.time_from_start <= self.time_before_land:
                self.publish_offboard_control_heartbeat_signal1()
            elif self.time_from_start > self.time_before_land:
                self.publish_offboard_control_heartbeat_signal2()


            if self.offboard_setpoint_counter == 10:
                self.engage_offboard_mode()
                self.arm()
                print("armed and switched to offboard mode")
            if self.offboard_setpoint_counter < 11:
                self.offboard_setpoint_counter += 1

        else:
            print(f"Offboard Callback: RC Flight Mode Channel {self.mode_channel} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
            self.offboard_setpoint_counter = 0



    def controller_timer_callback(self) -> None: # ~~This is the main function that runs at 100Hz and Administrates Calls to Every Other Function ~~
        print("Controller Callback")
        if self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard deciding whether to send heartbeat signal, engage offboard, and arm
            # self.time_from_start = time.time()-self.T0 #update curent time from start of program for reference trajectories and for switching between my controller and landing mode
            
            print(f"--------------------------------------")
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                print("IN OFFBOARD MODE")
                print(f"Controller callback- timefromstart: {self.time_from_start}")
                
                if self.time_from_start <= self.time_before_land: # our controller for first {self.time_before_land} seconds
                    print(f"Entering MPC Control Loop for next: {self.time_before_land-self.time_from_start} seconds")
                    self.controller()

                    state_input_ref_log_info = [
                        float(self.x), float(self.y), float(self.z), float(self.yaw),
                        float(self.final[0]), float(self.final[1]), float(self.final[2]), float(self.final[3]),
                        float(self.reffunc[0][-1]), float(self.reffunc[1][-1]), float(self.reffunc[2][-1]), float(self.reffunc[-1][-1]),
                        self.time_from_start, self.controller_callback_time, float(self.vx), float(self.vy), float(self.vz), 
                        float(self.roll), float(self.pitch), float(self.p), float(self.q), float(self.r), float(self.mpc_timel)
                    ]
                    self.update_logged_data(state_input_ref_log_info)

                elif self.time_from_start > self.time_before_land: #then land at origin and disarm
                    print("BACK TO SPAWN")
                    self.publish_position_setpoint(self.fin_pos[0], self.fin_pos[1], -0.3)
                    print(f"self.x: {self.x}, self.y: {self.y}, self.z: {self.z}")
                    if abs(self.x - self.fin_pos[0]) < 0.1 and abs(self.y - self.fin_pos[1]) < 0.1 and abs(self.z) <= 0.50:
                        print("Switching to Land Mode")
                        self.land()

                    state_input_ref_log_info = [
                        float(self.x), float(self.y), float(self.z), float(self.yaw),
                        float(self.final[0]), float(self.final[1]), float(self.final[2]), float(self.final[3]),
                        float(self.reffunc[0][-1]), float(self.reffunc[1][-1]), float(self.reffunc[2][-1]), float(self.reffunc[-1][-1]),
                        self.time_from_start, self.controller_callback_time, float(self.vx), float(self.vy), float(self.vz), 
                        float(self.roll), float(self.pitch), float(self.p), float(self.q), float(self.r), float(self.mpc_timel)
                    ]
                    self.update_logged_data(state_input_ref_log_info)

            if self.time_from_start > self.time_before_land:
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                    print("IN LAND MODE")
                    if abs(self.z) <= .15:
                        print("\nDisarming and Exiting Program")
                        self.disarm()
                        print("\nSaving all data!")
                        # if self.pyjoules_on:
                        #     self.csv_handler.save_data()
                        exit(0)

                state_input_ref_log_info = [
                        float(self.x), float(self.y), float(self.z), float(self.yaw),
                        float(self.final[0]), float(self.final[1]), float(self.final[2]), float(self.final[3]),
                        float(self.reffunc[0][-1]), float(self.reffunc[1][-1]), float(self.reffunc[2][-1]), float(self.reffunc[-1][-1]),
                        self.time_from_start, self.controller_callback_time, float(self.vx), float(self.vy), float(self.vz), 
                        float(self.roll), float(self.pitch), float(self.p), float(self.q), float(self.r), float(self.mpc_timel)
                    ]
                self.update_logged_data(state_input_ref_log_info)

            print(f"--------------------------------------")
            print("\n\n")
        else:
            print(f"Controller Callback: RC Flight Mode Channel {self.mode_channel} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")

    
    def controller(self): # Runs Algorithm Structure
        t0 = time.time()
        """MPC Controller Function."""
        print(f"NR States: {self.nr_state}") #prints current state

        if self.first_iteration:
            print("First Iteration")
            self.T0 = time.time()
            self.first_iteration = False

        self.time_from_start = time.time()-self.T0


#~~~~~~~~~~~~~~~ Calculate reference trajectory ~~~~~~~~~~~~~~~
        if self.time_from_start <= self.cushion_time:
            self.reffunc = self.hover_ref_func0()
        elif self.cushion_time < self.time_from_start < self.cushion_time + self.flight_time:
        # if self.time_from_start < self.flight_time:
            self.reffunc, self.fin_pos = self.ref_fun()

        # elif self.cushion_time + self.flight_time <= self.time_from_start <= self.time_before_land:
        elif self.flight_time <= self.time_from_start <= self.time_before_land:
            self.reffunc = self.hover_ref_func(self.fin_pos)
        else:
            self.reffunc = self.hover_ref_func(self.fin_pos)

        print(f"reffunc: {self.reffunc[:,0]}")

        # Calculate the MPC control input and transform the force into a throttle command for publishing to the vehicle
        new_u = self.get_new_control_input(self.reffunc)
        print(f"new_u: {new_u}")
        print(f"self.z: {self.z}")
        new_force = -new_u[0]       
        print(f"new_force: {new_force}")
        # exit(0)
        new_throttle = self.get_throttle_command_from_force(new_force)
        torque_x = new_u[1]
        torque_y = new_u[2]
        torque_z = new_u[3]

                                
        # Build the final input vector to save as self.u0 and publish to the vehicle via publish_rates_setpoint:
        self.final = [new_throttle, torque_x, torque_y, torque_z]
        # final = np.clip(final, 0.0, 1.0)
        current_input_save = np.array(self.final).reshape(-1, 1)
        print(f"newInput: \n{current_input_save}")
        self.u0 = current_input_save

        # Publish the final input to the vehicle
        self.publish_thrust_torque_setpoint(self.final[0], self.final[1], self.final[2], self.final[3])

        # Log the states, inputs, and reference trajectories for data analysis
        print(f"reffunc:({self.reffunc[0][-1], self.reffunc[1][-1], self.reffunc[2][-1], self.reffunc[-1][-1]})\n")
        self.controller_callback_time = time.time() - t0


# ~~ The following functions handle the log update and data retrieval for analysis ~~
    def update_logged_data(self, data):
        self.x_log.append(data[0])
        self.y_log.append(data[1])
        self.z_log.append(data[2])
        self.yaw_log.append(data[3])
        self.throttle_log.append(data[4])
        self.p_u_log.append(data[5])
        self.q_u_log.append(data[6])
        self.r_u_log.append(data[7])
        self.x_ref_log.append(data[8])
        self.y_ref_log.append(data[9])
        self.z_ref_log.append(data[10])
        self.yaw_ref_log.append(data[11])
        self.ctrl_loop_time_log.append(data[12])
        self.ctrl_callback_timel_log.append(data[13])
        self.vx_log.append(data[14])
        self.vy_log.append(data[15])
        self.vz_log.append(data[16])
        self.roll_log.append(data[17])
        self.pitch_log.append(data[18])
        self.p_log.append(data[19])
        self.q_log.append(data[20])
        self.r_log.append(data[21])
        self.mpc_timel_array_log.append(data[22])


    def get_x_log(self): return np.array(self.x_log).reshape(-1, 1)
    def get_y_log(self): return np.array(self.y_log).reshape(-1, 1)
    def get_z_log(self): return np.array(self.z_log).reshape(-1, 1)
    def get_vx_log(self): return np.array(self.vx_log).reshape(-1, 1)
    def get_vy_log(self): return np.array(self.vy_log).reshape(-1, 1)
    def get_vz_log(self): return np.array(self.vz_log).reshape(-1, 1)
    def get_roll_log(self): return np.array(self.roll_log).reshape(-1, 1)
    def get_pitch_log(self): return np.array(self.pitch_log).reshape(-1, 1)
    def get_yaw_log(self): return np.array(self.yaw_log).reshape(-1, 1)
    def get_p_log(self): return np.array(self.p_log).reshape(-1, 1)
    def get_q_log(self): return np.array(self.q_log).reshape(-1, 1)
    def get_r_log(self): return np.array(self.r_log).reshape(-1, 1)
    def get_ctrl_loop_time_log(self): return np.array(self.ctrl_loop_time_log).reshape(-1, 1)
    def get_x_ref_log(self): return np.array(self.x_ref_log).reshape(-1, 1)
    def get_y_ref_log(self): return np.array(self.y_ref_log).reshape(-1, 1)
    def get_z_ref_log(self): return np.array(self.z_ref_log).reshape(-1, 1)
    def get_yaw_ref_log(self): return np.array(self.yaw_ref_log).reshape(-1, 1)

    def get_throttle_log(self): return np.array(self.throttle_log).reshape(-1, 1)
    def get_p_u_log(self): return np.array(self.p_u_log).reshape(-1, 1)
    def get_q_u_log(self): return np.array(self.q_u_log).reshape(-1, 1)
    def get_r_u_log(self): return np.array(self.r_u_log).reshape(-1, 1)
    def get_ctrl_callback_timel_log(self): return np.array(self.ctrl_callback_timel_log).reshape(-1, 1)
    def get_mpc_timel_log(self): return np.array(self.mpc_timel_array_log).reshape(-1, 1)

    def get_new_control_input(self, reffunc):
        if self.pyjoules_on:
            with EnergyContext(handler=self.csv_handler, domains=[RaplPackageDomain(0), RaplCoreDomain(0)]):
                return self.execute_control_input(reffunc)
        else:
            return self.execute_control_input(reffunc)

    def ned2enu(self, ned): 
        """Convert NED coordinates to ENU."""
        x_ned, y_ned, z_ned = ned
        x_enu = y_ned
        y_enu = x_ned
        z_enu = -z_ned
        return np.array([x_enu, y_enu, z_enu])

    def execute_control_input(self, reffunc): 
        t0 = time.time()
        x0_nn = self.state_vector.reshape(-1)
        ref_now = reffunc[:,0]
        pos_x0 = x0_nn[0:3]
        pos_ref = ref_now[0:3]
        vel_x0 = x0_nn[6:9]
        agvel_x0 = x0_nn[9:12]
        p0_nn = np.concatenate([pos_x0 - pos_ref, np.array([x0_nn[3], x0_nn[4], x0_nn[5]]), np.array([ref_now[-1]]), vel_x0, agvel_x0], axis=0)
        if self.select_ctrl == 0:
            e0_nn = (torch.tensor(p0_nn, device=self.device, dtype=torch.float32).reshape(1, -1) - self.d_mean) / self.d_std
            U_opt_nn = self.nn_controller(e0_nn)
            U_opt_nn = U_opt_nn * (self.d_max - self.d_min) + self.d_min
            u_mpc = U_opt_nn.reshape(-1, 1).detach().cpu().numpy()
        if self.select_ctrl == 1:
            e0_nn = (p0_nn - self.d_mean) / self.d_std
            U_opt_nn = self.forward_numpy(e0_nn)
            U_opt_nn = U_opt_nn * (self.d_max - self.d_min) + self.d_min
            u_mpc = U_opt_nn.reshape(-1, 1)
        
        self.mpc_timel = time.time() - t0
        print(f"Outside Acados Timel: {self.mpc_timel}sec, Good Enough for {1/self.mpc_timel}Hz")
        # print(f"status: {status}")
        # print(f"x_mpc: {x_mpc}")
        print(f"u_mpc: {u_mpc}")
        return u_mpc


# ~~ The following functions are reference trajectories for tracking ~~
    def hover_ref_func0(self): 
        """ Returns constant hover reference trajectories at a few different positions. """
        
        r = np.array([0.0, 0.0, -1.0, 0.0])
        r_final = np.tile(r, (self.num_steps, 1 )).T
        return r_final

    def hover_ref_func(self, final_pos): #Returns Constant Hover Reference Trajectories At A Few Different Positions for Testing ([x,y,z,yaw])
        """ Returns constant hover reference trajectories at a few different positions. """
          
        r = final_pos
        r_final = np.tile(r, (self.num_steps, 1 )).T
        return r_final
    

    def circle_horz_ref_func(self):
        """ Returns circle reference trajectory in horizontal plane. """
        print("circle_horz_ref_func")

        # Generate a time array for the trajectory
        t_start = self.time_from_start - self.cushion_time
        # t = np.linspace(t_start, t_start + self.horizon, self.num_steps)
        t = t_start

        PERIOD = 12.0

        if self.double_speed:
            PERIOD /= 2

        w = 2 * np.pi / PERIOD

        # Compute trajectory components as arrays
        x = 1.5 * np.cos(w * t ) - 1.0  
        y = 1.5 * np.sin(w * t )  
        z = -1.0  
        yaw = 0.0  

        # Construct the reference trajectory array
        r = np.array([x, y, z, yaw])
        rr = np.tile(r, (self.num_steps, 1 )).T

        x_final = 1.5 * np.cos(w * self.flight_time ) - 1.0
        y_final = 1.5 * np.sin(w * self.flight_time )
        z_final = -1.0 
        yaw_final = 0.0
        final_pos = np.array([x_final, y_final, z_final, yaw_final])

        return rr, final_pos

    

    
    def fig8_horz_ref_func(self):
        """ Returns figure 8 reference trajectory in horizontal plane. """
        print("fig8_horz_ref_func")

        # Generate a time array for the trajectory
        t_start = self.time_from_start - self.cushion_time
        t = t_start

        PERIOD = 12.0

        if self.double_speed:
            PERIOD /= 2

        w = 2 * np.pi / PERIOD

        # Compute trajectory components as arrays
        x = 1.5 * np.sin(2 * w * t) 
        y = 1.5 * np.sin(w * t) 
        z = -1.0 
        yaw = 0.0


        r = np.array([x, y, z, yaw])
        rr = np.tile(r, (self.num_steps, 1 )).T

        x_final = 1.5 * np.sin(2 * w * self.flight_time) 
        y_final = 1.5 * np.sin(w * self.flight_time) 
        z_final = -1.0 
        yaw_final = 0.0
        final_pos = np.array([x_final, y_final, z_final, yaw_final])


        return rr, final_pos

    def helix(self):
        """ Returns helix reference trajectory. """
        print("helix")
        
        # Generate a time array for the trajectory
        t_start = self.time_from_start - self.cushion_time
        t = t_start

        PERIOD = 12.0

        if self.double_speed:
            PERIOD /= 2

        w = 2 * np.pi / PERIOD


        # Compute trajectory components as arrays
        x = 1.5 * np.cos(w * t) - 1.5
        y = 1.5 * np.sin(w * t)
        z = -1 * (1.0 + 0.07*t)

        yaw = 0.0

        r = np.array([x, y, z, yaw])
        rr = np.tile(r, (self.num_steps, 1 )).T

        x_final = 1.5 * np.cos(w * self.flight_time) - 1.5
        y_final = 1.5 * np.sin(w * self.flight_time)
        z_final = -1 * (1.0 + 0.07*self.flight_time)
        yaw_final = 0.0
        final_pos = np.array([x_final, y_final, z_final, yaw_final])

        return rr, final_pos
    
    def ssinn(self):
        """ Returns helix reference trajectory. """
        print("helix")
        
        # Generate a time array for the trajectory
        t_start = self.time_from_start - self.cushion_time
        t = t_start

        PERIOD = 12.0

        if self.double_speed:
            PERIOD /= 2

        w = 2 * np.pi / PERIOD


        # Compute trajectory components as arrays
        x = (3.0/12.0)*t - 1.5
        y = 1.5 * np.sin(w * t)
        z = -1.0 
        yaw = 0.0

        r = np.array([x, y, z, yaw])
        rr = np.tile(r, (self.num_steps, 1 )).T


        x_final = (3.0/12.0)*self.flight_time - 1.5
        y_final = 1.5 * np.sin(w * self.flight_time)
        z_final = -1.0 
        yaw_final = 0.0
        final_pos = np.array([x_final, y_final, z_final, yaw_final])

        return rr, final_pos

    def triangle_traj(self):

        t_start = self.time_from_start - self.cushion_time
        t = t_start

        PERIOD = 12.0

        if self.double_speed:
            PERIOD /= 2

  
        P1 = np.array([1.5, 0.0, -1.0])
        P2 = np.array([0.0, 1.5 * np.sqrt(3), -2.0])
        P3 = np.array([-1.5, 0.0, -1.0])

        tau = t % PERIOD

        if tau < 4.0:
            s = tau / 4.0
            p = P1 + s * (P2 - P1)

        elif tau < 8.0:
            s = (tau - 4.0) / 4.0
            p = P2 + s * (P3 - P2)

        else:
            s = (tau - 8.0) / 4.0
            p = P3 + s * (P1 - P3)

        yaw = 0.0

        r = np.array([p[0], p[1], p[2], yaw])
        rr = np.tile(r, (self.num_steps, 1 )).T

        tau_final = self.flight_time % PERIOD
        if tau_final < 4.0:
            s_final = tau_final / 4.0
            p_final = P1 + s_final * (P2 - P1)
        elif tau_final < 8.0:
            s_final = (tau_final - 4.0) / 4.0
            p_final = P2 + s_final * (P3 - P2)
        else:
            s_final = (tau_final - 8.0) / 4.0
            p_final = P3 + s_final * (P1 - P3)
        yaw_final = 0.0
        final_pos = np.array([p_final[0], p_final[1], p_final[2], yaw_final])

        return rr, final_pos


    


# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main(args=None):
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    logger = None

    def shutdown_logging(*args):
        print("\nInterrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...")
        if logger:
            logger.log(offboard_control)
        offboard_control.destroy_node()
        rclpy.shutdown()
    # Register the signal handler for Ctrl+C (SIGINT)
    # signal.signal(signal.SIGINT, shutdown_logging)

    try:
        print(f"\nInitializing ROS 2 node: '{__name__}' for offboard control")
        logger = Logger([sys.argv[1]])  # Create logger with passed filename
        rclpy.spin(offboard_control)    # Spin the ROS 2 node
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected (Ctrl+C), exiting...")
    except Exception as e:
        print(f"\nError in main: {e}")
        traceback.print_exc()
    finally:
        shutdown_logging()
        if offboard_control.pyjoules_on:
            offboard_control.csv_handler.save_data()
        print("\nNode has shut down.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError in __main__: {e}")
        traceback.print_exc()


