import numpy as np
import matplotlib.pyplot as plt



MOTOR_INPUT_SCALING = 2870.159048
MOTOR_VELOCITY_ARMED = 100.0
MOTOR_CONSTANT = 0.000002699265381
# collective_thrust = np.linspace(0, 1.526*9.79136, 100)  

# motor_speed = np.sqrt(collective_thrust / (MOTOR_CONSTANT))
# throttle_command = (motor_speed - MOTOR_VELOCITY_ARMED) / MOTOR_INPUT_SCALING

throttle_command = np.linspace(0, 1, 11)  # Throttle command from 0 to 1                                                                                                                                
                                                                                                                                                                                                                                   
collective_thrust = (throttle_command * MOTOR_INPUT_SCALING + MOTOR_VELOCITY_ARMED) ** 2 * MOTOR_CONSTANT
throttle = np.linspace(0, 1, 11)  
thrust = 2.815301 * throttle ** 2 + 10.695885 * throttle - 0.243072
# a = 19.2463167420814
# b = 41.8467162352942
# c = -7.19353022443441
# thrust = a * throttle ** 2 + b * throttle + c


plt.figure()
plt.plot(collective_thrust, throttle_command)
plt.plot(thrust, throttle, 'r--', label='F60 Pro Throttle Curve')
plt.xlabel('Collective Thrust (N)')
plt.ylabel('Throttle Command')
plt.title('Throttle Command vs Collective Thrust')
plt.grid(True)
plt.show()




