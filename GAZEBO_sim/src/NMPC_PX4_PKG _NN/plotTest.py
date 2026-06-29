import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Load the data from the CSV file
    data = np.genfromtxt('/<your_ros2_ws>/build/nmpc_px4_pkg/nmpc_px4_pkg/data_analysis/logs/log.log', delimiter=',', skip_header=1)

    # Extract columns (adjust indices based on your CSV structure)
    time = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    z = -data[:, 3]
    vx = data[:, 4]
    vy = data[:, 5]
    vz = data[:, 6]
    roll = data[:, 7]*180.0/np.pi
    pitch = data[:, 8]*180.0/np.pi
    yaw = data[:, 9]*180.0/np.pi
    p = data[:, 10]*180.0/np.pi
    q = data[:, 11]*180.0/np.pi
    r = data[:, 12]*180.0/np.pi
    x_ref = data[:, 13]
    y_ref = data[:, 14]
    z_ref = -data[:, 15]
    yaw_ref = data[:, 16]*180.0/np.pi

    throttle = data[:, 17]
    mx = data[:, 18]
    my = data[:, 19]
    mz = data[:, 20]

    sol_time = data[:, 21]

    print(x.shape)

    plt.figure()
    plt.plot(time, sol_time, label='sol_time')
    plt.title('sol_time Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('sol_time (s)')
    plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.plot(x_ref, y_ref, z_ref, 'r--')
    ax.set_title('3D Trajectory')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')

    # Plotting
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(time, x, label='x')
    plt.plot(time, x_ref, label='x_ref', linestyle='--')
    plt.title('X Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position (m)')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(time, y, label='y')
    plt.plot(time, y_ref, label='y_ref', linestyle='--')
    plt.title('Y Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(time, z, label='z')
    plt.plot(time, z_ref, label='z_ref', linestyle='--')
    plt.title('Z Position vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(time, yaw, label='yaw')
    plt.plot(time, yaw_ref, label='yaw_ref', linestyle='--')
    plt.title('Yaw vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Yaw (rad)')
    plt.legend()

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(time, roll, label='x')
    plt.title('roll vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('roll (deg)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(time, pitch, label='pitch')
    plt.title('Pitch vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (deg)')
    plt.legend()

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(time, throttle, label='x')
    plt.title('throttle vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('throttle ')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(time, mx, label='y')
    plt.title('mx vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('mx (Nm)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(time, my, label='z')
    plt.title('my vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('my (Nm)')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(time, mz, label='yaw')
    plt.title('mz vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('mz (Nm)')
    plt.legend()


    plt.figure()

    plt.subplot(2, 3, 1)
    plt.plot(time, p, label='x')
    plt.title('p vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('p (deg/s)')
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(time, q, label='y')
    plt.title('q vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('q (deg/s)')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(time, r, label='z')
    plt.title('r vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('r (deg/s)')
    plt.legend()



    plt.subplot(2, 3, 4)
    plt.plot(time, vx, label='x')
    plt.title('vx vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('vx (m/s)')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(time, vy, label='y')
    plt.title('vy vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('vy (m/s)')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    plt.plot(time, vz, label='z')
    plt.title('vz vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('vz (m/s)')
    plt.legend()
    



    plt.tight_layout()
    plt.show()


