import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.parameter import Parameter

class EKF(Node):
    def __init__(self):
        super().__init__('EKF')
        self.F = np.eye(4) #State Transition Matrix
        self.R_wb = np.eye(3) #Rotation Matrix of IMU frame w.r.t World frame
        self.x = np.array([[1],[0],[0],[0]]) #Initializing the state variables qw,qx,qy,qz
        self.E = np.eye(4)*0.1 #State Covariance Matrix
        self.Q = np.eye(3)*0.1 #Gyroscope noise covariance matrix
        self.N = np.eye(3)*1 #Accelerometer noise covariance matrix
        self.gyro = np.zeros((3,1)) #Gyroscope readings
        self.accel = np.zeros((3,1)) #Accelerometer readings
        self.dt = 0.01 #Callback time step
        self.initialize = False #EKF system initialization flag
        self.previous_time = 0 #Record previous sensor message time-stamp, used to calculate dt
        self.IMU_sub = self.create_subscription(
            Imu,
            '/camera/imu',
            self.IMUcallback,
            10
        ) #IMU callback function

    #Returns Rotation Matrix R of IMU body w.r.t world.
    def getRotationMatfromQ(self,q):
        qw, qx, qy, qz = q[0][0],q[1][0],q[2][0],q[3][0]
        R = np.array([[1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]])
        return R

    def getQuaternionDerivative(self,q,omega):

        qw, qx, qy, qz = q[0][0],q[1][0],q[2][0],q[3][0]
        wx, wy, wz = omega[0][0],omega[1][0],omega[2][0]

        qw_dot = -0.5 * (qx * wx + qy * wy + qz * wz)
        qx_dot = 0.5 * (qw * wx + qy * wz - qz * wy)
        qy_dot = 0.5 * (qw * wy + qz * wx - qx * wz)
        qz_dot = 0.5 * (qw * wz + qx * wy - qy * wx)

        return np.array([[qw_dot], [qx_dot], [qy_dot], [qz_dot]])
    
    #Returns Jacobian of state transition matrix
    def getA(self,omega_roll,omega_pitch,omega_yaw):
        wx, wy, wz = omega_roll,omega_pitch,omega_yaw
        A = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])

        return 0.5 * A
    
    def getV(self,q):
        w, x, y, z = q[0][0],q[1][0],q[2][0],q[3][0]
        V = np.array([[-x,-y,-z],[w,-z,y],[z,w,-x],[-y,x,w]])
        return 0.5*V
    #Returns Jacobian of Observation model
    def getH(self, q,a_x,a_y,a_z):
        qw, qx, qy, qz = q[0][0],q[1][0],q[2][0],q[3][0]

        H = np.array([[2*a_z*qy-2*a_y*qz, 2*a_y*qy + 2*a_z*qz, 2*a_y*qx-4*a_x*qy+2*a_z*qw,2*a_z*qx-4*a_x*qz-2*a_y*qw],[2*a_x*qz - 2*a_z*qx, 2*a_x*qy-4*a_y*qx-2*a_z*qw,2*a_x*qx+2*a_z*qz, 2*a_x*qw-4*a_y*qz+2*a_z*qy],[2*a_y*qx-2*a_x*qy, 2*a_y*qw+2*a_x*qz-4*a_z*qx, 2*a_y*qz-2*a_x*qw-4*a_z*qy,2*a_x*qx+2*a_y*qy]])
        return H

    #Prediction step of the EKF. Calculates and returns the predicted state and state covariance matrix, given Gyroscope readings
    def predict(self, x, E,omega,dt):
        x_dot = self.getQuaternionDerivative(x,omega)
        x_pred = x + x_dot*dt
        x_norm = np.linalg.norm(x_pred)
        x_pred = x_pred/x_norm
        A = self.getA(omega[0][0],omega[1][0],omega[2][0])
        F = np.eye(4) + dt*A
        V = self.getV(x_pred)
        Q = self.Q*dt
        E_pred = F.dot(E).dot(F.T) + V.dot(Q).dot(V.T)
        return x_pred, E_pred
    #Update step of the EKF. Calculates and returns the updated state and state covariance matrix, given Accelerometer readings
    def update(self, x, E, z):
        H = self.getH(x,z[0][0],z[1][0],z[2][0])
        R = self.getRotationMatfromQ(x)
        W = R
        g = np.array([[0],[0],[+9.81]])
        r = R.dot(z) - g
        temp = H.dot(E).dot(H.T) + W.dot(self.N).dot(W.T)
        P_inv = np.linalg.inv(temp)
        K = E.dot(H.T).dot(P_inv)
        x_upd = x - K.dot(r)
        x_norm = np.linalg.norm(x_upd)
        x_upd = x_upd/x_norm
        E_upd = E - K.dot(H).dot(E)
        return x_upd, E_upd
    
    def quaternion_to_euler(self,q):
        qw, qx, qy, qz = q[0][0],q[1][0],q[2][0],q[3][0]

        # Roll (x-axis rotation)
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

        # Pitch (y-axis rotation)
        pitch = np.arcsin(2 * (qw * qy - qz * qx))

        # Yaw (z-axis rotation)
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        return np.array([roll, pitch, yaw])
    
    #Subscribes to IMU messages and performs the EKF algorithm
    def IMUcallback(self,msg):
        if(self.initialize == False):
            self.previous_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.initialize = True
            return
        
        # Extract gyroscope readings
        gyroscope_x = msg.angular_velocity.x
        gyroscope_y = msg.angular_velocity.y
        gyroscope_z = msg.angular_velocity.z

        # Extract accelerometer readings
        accelerometer_x = msg.linear_acceleration.x
        accelerometer_y = msg.linear_acceleration.y
        accelerometer_z = msg.linear_acceleration.z
        # Express this in the ROS convention (x->forward, y->left, z->up)
        gyro_ros = np.array([[gyroscope_z],[-gyroscope_x],[-gyroscope_y]])
        accel_ros = np.array([[accelerometer_z],[-accelerometer_x],[-accelerometer_y]])
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Calculate dt used in the EKF algorithm
        self.dt = current_time  - self.previous_time
        #Perform the prediction and update steps of the EKF algorithm
        x_pred, E_pred = self.predict(self.x,self.E,gyro_ros,self.dt)
        x_upd, E_upd = self.update(x_pred,E_pred,accel_ros)
        #Update the state x and the state covariance matrix E
    
        self.x = x_upd
        self.E = E_upd
        roll,pitch,yaw = self.quaternion_to_euler(self.x)
        #print(f"Current state is: {self.x}")
        #print(f"Current covariance is: {E_upd}")
        print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
        self.previous_time = current_time
        return


def main():
    rclpy.init()
    node = EKF()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
