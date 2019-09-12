from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio,
                 max_lat_accel, max_steer_angle):
        # Set up Controller for steering
        min_speed = 0.1
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        # Set up PID controller for throttle
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0. # minimum throttle value
        mx = 0.2 # maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        # Set up LowPassFilter for noisly incoming velocity
        tau = 0.5 # 1/(2*tau*pi) = cutoff frequency
        ts = 0.02 # sample_time to 50 Hz
        self.vel_lpf = LowPassFilter(tau, ts)

        # Store relevant variables for throttle and
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.min_speed = min_speed
        # Store initial timestamp
        self.last_time = rospy.get_time()


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        """Function that calculates the throttle, brake and steering according to
        incoming current_vel, dbw_enabled, linear_vel and angular_vel when the dbw is enabled
		otherwise it resets the throttle_controller"""
        if not dbw_enabled:
            self.throttle_controller.reset()
            self.vel_lpf.ready = False
            return 0.,0.,0.

        # Apply LowPassFilter to filter out high frequencies in velocity
        current_vel = self.vel_lpf.filt(current_vel)

        # Calc steering
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        # Calc error in velocity
        vel_error = linear_vel - current_vel
        #self.last_vel = current_vel

        # Calc throttle and brake values
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0.
        # Check if car should stand or brake, otherwise accel as calced above
        if linear_vel == 0. and current_vel < self.min_speed:
            throttle = 0.
            brake = 700 #Nm -> Force to hold vehicle in place if stopped at red light
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0.
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque in Nm

        return throttle, brake, steering
