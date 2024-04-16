import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.parameter import Parameter

class EKF(Node):
    def __init__(self):
        super().__init__('EKF')
        self.F = np.eye(3) #State Transition Matrix
        self.R_wb = np.eye(3) #Rotation Matrix of IMU frame w.r.t World frame
        self.x = np.zeros((3,1)) #Initializing the state variables roll,pitch,yaw
        self.E = np.eye(3)*0.5 #State Covariance Matrix
        self.Q = np.eye(3)*0.9 #Gyroscope noise covariance matrix
        self.N = np.eye(3)*0.1 #Accelerometer noise covariance matrix
        self.gyro = np.zeros((3,1)) #Gyroscope readings
        self.accel = np.zeros((3,1)) #Accelerometer readings
        self.dt = 0.01 #Callback time step
        self.initialize = False #EKF system initialization flag
        self.use_rk4 = False #Set true to use Runge-Kutta fourth order integration method for state propagation, else set false to use Euler integration
        self.previous_time = 0 #Record previous sensor message time-stamp, used to calculate dt
        self.IMU_sub = self.create_subscription(
            Imu,
            '/camera/imu',
            self.IMUcallback,
            10
        ) #IMU callback function

    #Returns Rotation Matrix R of IMU body w.r.t world.
    def getRotationMatRPY(self,roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],[0, np.cos(roll), -np.sin(roll)],[0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],[0,1,0],[-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        R = np.dot(R_z,np.dot(R_y,R_x))
        return R
    #Returns Jacobian of state transition matrix
    def getA(self,roll,pitch,yaw,omega_roll,omega_pitch,omega_yaw):
        sigma1 = omega_yaw*np.cos(pitch)*np.cos(roll) - omega_roll*np.sin(pitch) + omega_pitch*np.cos(pitch)*np.sin(roll)
        sigma2 = np.cos(yaw)*np.sin(roll) - np.cos(roll)*np.sin(pitch)*np.sin(yaw)
        sigma3 = np.cos(roll)*np.sin(yaw) - np.cos(yaw)*np.sin(pitch)*np.sin(roll)
        sigma4 = np.sin(roll)*np.sin(yaw) + np.cos(roll)*np.cos(yaw)*np.sin(pitch)
        sigma5 = np.cos(roll)*np.cos(yaw) + np.sin(pitch)*np.sin(roll)*np.sin(yaw)
        A = np.array([[omega_yaw*sigma3+omega_pitch*sigma4, np.cos(yaw)*sigma1, omega_yaw*sigma2-omega_pitch*sigma5-omega_roll*np.cos(pitch)*np.sin(yaw)],[-omega_yaw*sigma5-omega_pitch*sigma2, np.sin(yaw)*sigma1,omega_yaw*sigma4-omega_pitch*sigma3+omega_roll*np.cos(pitch)*np.cos(yaw)],[np.cos(pitch)*(omega_pitch*np.cos(roll)-omega_yaw*np.sin(roll)), -omega_roll*np.cos(pitch)-omega_yaw*np.cos(roll)*np.sin(pitch)-omega_pitch*np.sin(pitch)*np.sin(roll), 0]])
        return A
    #Returns Jacobian of Observation model
    def getH(self, roll,pitch,yaw,a_x,a_y,a_z):
        sigma1 = a_z*np.cos(pitch)*np.cos(roll) - a_x*np.sin(pitch) + a_y*np.cos(pitch)*np.sin(roll)
        sigma2 = np.cos(yaw)*np.sin(roll) - np.cos(roll)*np.sin(pitch)*np.sin(roll)
        sigma3 = np.cos(roll)*np.sin(yaw) - np.cos(yaw)*np.cos(pitch)*np.sin(roll)
        sigma4 = np.sin(roll)*np.sin(yaw) + np.cos(roll)*np.cos(yaw)*np.sin(pitch)
        sigma5 = np.cos(roll)*np.cos(yaw) + np.sin(pitch)*np.sin(roll)*np.sin(yaw)

        H = np.array([[a_y*sigma4+a_z*sigma3, np.cos(yaw)*sigma1, a_z*sigma2 - a_y*sigma5 - a_x*np.cos(pitch)*np.sin(yaw)],[-a_y*sigma2 - a_z*sigma5, np.sin(yaw)*sigma1, a_z*sigma4 - a_y*sigma3 + a_x*np.cos(pitch)*np.cos(yaw)],[np.cos(pitch)*(a_y*np.cos(roll)-a_z*np.sin(roll)), -a_x*np.cos(pitch)-a_z*np.cos(roll)*np.sin(pitch) - a_y*np.sin(pitch)*np.sin(roll), 0]])
        return H
    #Performs Runge-Kutta fourth order integration and returns the predicted state
    def RK4(self,x,omega,dt):
        R1 = self.getRotationMatRPY(x[0][0],x[1][0],x[2][0])
        X1 = R1.dot(omega)
        R2 = self.getRotationMatRPY(x[0][0]+X1[0][0]*dt/2,x[1][0]+X1[1][0]*dt/2,x[2][0]+X1[2][0]*dt/2)
        X2 = R2.dot(omega)
        R3 = self.getRotationMatRPY(x[0][0]+X2[0][0]*dt/2,x[1][0]+X2[1][0]*dt/2,x[2][0]+X2[2][0]*dt/2)
        X3 = R3.dot(omega)
        R4 = self.getRotationMatRPY(x[0][0]+X3[0][0]*dt,x[1][0]+X3[1][0]*dt,x[2][0]+X3[2][0]*dt)
        X4 = R4.dot(omega)
        x_pred = x + (X1 + 2*X2 + 2*X3 + X4)*dt/6
        return x_pred

    #Prediction step of the EKF. Calculates and returns the predicted state and state covariance matrix, given Gyroscope readings
    def predict(self, x, E,omega,dt):
        if(self.use_rk4):
            x_pred = self.RK4(x,omega,dt)
        else:
            R1 = self.getRotationMatRPY(x[0][0],x[1][0],x[2][0])
            x_pred = x + R1.dot(omega)*dt
        R = self.getRotationMatRPY(x_pred[0][0],x_pred[1][0],x_pred[2][0])
        A = self.getA(x_pred[0][0],x_pred[1][0],x_pred[2][0],omega[0][0],omega[1][0],omega[2][0])
        F = np.eye(3) + dt*A
        V = -R
        Q = self.Q*dt
        E_pred = F.dot(E).dot(F.T) + V.dot(Q).dot(V.T)
        return x_pred, E_pred
    #Update step of the EKF. Calculates and returns the updated state and state covariance matrix, given Accelerometer readings
    def update(self, x, E, z):
        H = self.getH(x[0][0],x[1][0],x[2][0],z[0][0],z[1][0],z[2][0])
        R = self.getRotationMatRPY(x[0][0],x[1][0],x[2][0])
        g = np.array([[0],[0],[+9.81]])
        r = R.dot(z) - g
        temp = H.dot(E).dot(H.T) + R.dot(self.N).dot(R.T)
        P_inv = np.linalg.inv(temp)
        K = E.dot(H.T).dot(P_inv)
        x_upd = x - K.dot(r)
        E_upd = E - K.dot(H).dot(E)
        return x_upd, E_upd
    
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
        for i in range(3):
            if(x_upd[i][0]>=0):
                x_upd[i][0] %=3.14
            else:
                x_upd[i][0] = -1*(abs(x_upd[i][0])%3.14)

        self.x = x_upd
        self.E = E_upd
        print(f"Current state is: {self.x*(180/3.14)}")
        #print(f"Current covariance is: {E_upd}")
        #print(self.getA(self.x[0][0],self.x[1][0],self.x[2][0],gyro_ros[0][0],gyro_ros[1][0],gyro_ros[2][0]))
        #print(f"dt is: {self.dt}")
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












