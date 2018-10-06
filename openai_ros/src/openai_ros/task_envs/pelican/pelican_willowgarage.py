#!/usr/bin/env python

import rospy
import numpy
import random
import time
from gym import spaces
from openai_ros.robot_envs import pelican_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from .tf_utils import euler_from_quaternion
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from mav_msgs.msg import Actuators
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose

timestep_limit_per_episode = 400 # Can be any Value

register(
        id='PelicanNavWillowgarageEnv-v0',
        entry_point='openai_ros:task_envs.pelican.pelican_willowgarage.PelicanNavWillowgarageEnv',
        timestep_limit=timestep_limit_per_episode,
    )

class PelicanNavWillowgarageEnv(pelican_env.PelicanEnv):
    def __init__(self):
        """
        Make Pelican learn how to move straight from The starting point
        to a desired point inside the designed corridor.
        http://robotx.org/images/files/RobotX_2018_Task_Summary.pdf
        Demonstrate Navigation Control
        """
        
        # Only variable needed to be set here
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.logdebug("Start PelicanEnvTwoSetsBuoysEnv INIT...")
        self.action_space = spaces.Box(0, 1500, shape=(4,), dtype='float32')
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        
        
        # Actions and Observations
        self.propeller_high_speed = rospy.get_param('/pelican/propeller_high_speed')
        self.propeller_low_speed = rospy.get_param('/pelican/propeller_low_speed')
        self.max_angular_speed = rospy.get_param('/pelican/max_angular_speed')
        self.max_yaw_angular_speed = rospy.get_param('/pelican/max_yaw_angular_speed')
        self.max_distance_from_des_point = rospy.get_param('/pelican/max_distance_from_des_point')
        
        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/pelican/desired_point/x")
        self.desired_point.y = rospy.get_param("/pelican/desired_point/y")
        self.desired_point.z = rospy.get_param("/pelican/desired_point/z")
        self.desired_point_epsilon = rospy.get_param("/pelican/desired_point_epsilon")
        
        self.work_space_x_max = rospy.get_param("/pelican/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/pelican/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/pelican/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/pelican/work_space/y_min")
        self.work_space_z_max = rospy.get_param("/pelican/work_space/z_max")
        self.work_space_z_min = rospy.get_param("/pelican/work_space/z_min")
        
        self.dec_obs = rospy.get_param("/pelican/number_decimals_precision_obs")
        self.energy_cost_weight = rospy.get_param("/pelican/energy_cost_weight")
        self.propeller_hovering_speed = rospy.get_param("/pelican/propeller_hovering_speed")
        self.propeller_normalize_constant = rospy.get_param("/pelican/propeller_normalize_constant")
        self.attitude_cost_weight = rospy.get_param("/pelican/attitude_cost_weight")
        # We place the Maximum and minimum values of observations

        high = numpy.array([self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max,
                            1.57,
                            1.57,
                            3.14,
                            self.propeller_high_speed,
                            self.propeller_high_speed,
                            self.propeller_high_speed,
                            self.max_angular_speed,
                            self.max_angular_speed,
                            self.max_yaw_angular_speed,
                            self.max_distance_from_des_point,
                            # numpy.inf,
                            # numpy.inf,
                            # numpy.inf
                            ])
                                        
        low = numpy.array([ self.work_space_x_min,
                            self.work_space_y_min,
                            self.work_space_z_max,
                            -1*1.57,
                            -1*1.57,
                            -1*3.14,
                            -1*self.propeller_high_speed,
                            -1*self.propeller_high_speed,
                            -1*self.propeller_high_speed,
                            -1*self.max_angular_speed,
                            -1*self.max_angular_speed,
                            -1*self.max_yaw_angular_speed,
                            0.0,
                            # -numpy.inf,
                            # -numpy.inf,
                            # -numpy.inf
                            ])

        
        self.observation_space = spaces.Box(low, high)
        self.a_dim = 4
        self.s_dim = high.size
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        
        self.done_reward =rospy.get_param("/pelican/done_reward")
        self.closer_to_point_reward = rospy.get_param("/pelican/closer_to_point_reward")

        self.cumulated_steps = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(PelicanNavWillowgarageEnv, self).__init__()
        
        rospy.logdebug("END PelicanNavWillowgarageEnv INIT...")

    def _set_init_pose(self):
        """
        Sets the four proppelers speed to 0.0 and waits for the time_sleep
        to allow the action to be executed
        """

        print("begin to reset")
        # reset mav state to initial position
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            # s = self.set_model_state(self.initial_mav_state)
            init_position = Point()
            init_position.x = random.uniform(-5, 5)
            init_position.y = random.uniform(-5, 5)
            init_position.z = random.uniform(0.5, 10)
            s = self.set_model_state(ModelState('pelican', Pose(init_position, Quaternion(0.0, 0.0, 0.0, 1.0)),
                                                Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)), 'world'))
            # print("reset pelican position to (0, 0, 5)")
        except (rospy.ServiceException) as e:
            print("/gazebo/set_model_state service call failed")

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """

        # For Info Purposes
        self.cumulated_reward = 0.0
        # We get the initial pose to mesure the distance from the desired point.
        odom = self.get_odom()
        current_position = Vector3()
        current_position.x = odom.pose.pose.position.x
        current_position.y = odom.pose.pose.position.y
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(current_position)

        

    def _set_action(self, action):
        """
        It sets the joints of pelican based on the action integer given
        based on the action number given.
        :param action: The action integer that sets what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        motor_cmd = Actuators(angular_velocities=[0,0,0,0])
        for i in range(self.a_dim):
            motor_cmd.angular_velocities[i] = action[i]
        self._cmd_drive_pub.publish(motor_cmd)
        time.sleep(0.02)
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have access to, we need to read the
        PelicanEnv API DOCS.
        :return: observation
        """
        rospy.logdebug("Start Get Observation ==>")

        odom = self.get_odom()
        base_position = odom.pose.pose.position
        base_orientation_quat = odom.pose.pose.orientation
        base_roll, base_pitch, base_yaw = self.get_orientation_euler(base_orientation_quat)
        base_speed_linear = odom.twist.twist.linear
        base_speed_angular_yaw = odom.twist.twist.angular.z
        base_speed_angular_pitch = odom.twist.twist.angular.x
        base_speed_angular_roll = odom.twist.twist.angular.y
        
        distance_from_desired_point = self.get_distance_from_desired_point(base_position)

        observation = []
        observation.append(round(self.desired_point.x-base_position.x,self.dec_obs))
        observation.append(round(self.desired_point.y-base_position.y,self.dec_obs))
        observation.append(round(self.desired_point.z-base_position.z,self.dec_obs))
        
        observation.append(round(base_roll,self.dec_obs))
        observation.append(round(base_pitch,self.dec_obs))
        observation.append(round(base_yaw,self.dec_obs))
        
        observation.append(round(base_speed_linear.x,self.dec_obs))
        observation.append(round(base_speed_linear.y,self.dec_obs))
        observation.append(round(base_speed_linear.z,self.dec_obs))
        
        observation.append(round(base_speed_angular_pitch,self.dec_obs))
        observation.append(round(base_speed_angular_roll,self.dec_obs))
        observation.append(round(base_speed_angular_yaw,self.dec_obs))
        
        observation.append(round(distance_from_desired_point,self.dec_obs))
        # observation.append(round(self.desired_point.x,self.dec_obs))
        # observation.append(round(self.desired_point.y,self.dec_obs))
        # observation.append(round(self.desired_point.z,self.dec_obs))

        return observation
        

    def _is_done(self, observations):
        """
        We consider the episode done if:
        1) The pelican is ouside the workspace
        2) It got to the desired point
        """
        distance_from_desired_point = observations[12]

        current_position = Vector3()
        current_position.x = self.desired_point.x - observations[0]
        current_position.y = self.desired_point.y - observations[1]
        current_position.z = self.desired_point.z - observations[2]
        
        is_inside_corridor = self.is_inside_workspace(current_position)
        has_reached_des_point = self.is_in_desired_position(current_position, self.desired_point_epsilon)
        
        done = not(is_inside_corridor) or has_reached_des_point
        
        return done

    def _compute_reward(self, observations, action, done):
        """
        We Base the rewards in if its done or not and we base it on
        if the distance to the desired point has increased or not
        :return:
        """

        # We consider the 3D space distance
        current_position = Point()
        current_position.x = self.desired_point.x - observations[0]
        current_position.y = self.desired_point.y - observations[1]
        current_position.z = self.desired_point.z - observations[2]
        # print("======================================================================")
        # print("current position: " + str(observations[0]) + " " + str(observations[1]) + " " + str(observations[2]))
        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference =  distance_from_des_point - self.previous_distance_from_des_point




        if not done:
            
            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                #rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward = self.closer_to_point_reward
            else:
                #rospy.logerr("ENCREASE IN DISTANCE BAD")
                reward = -1*self.closer_to_point_reward
            
            # we punish energe waste
            normalize_action = action - numpy.array([self.propeller_hovering_speed] * self.a_dim)
            normalize_action = normalize_action/self.propeller_normalize_constant
            # print("action: {}".format(action))
            # print("normalize_action: {}".format(normalize_action))
            energy_cost = self.energy_cost_weight * numpy.linalg.norm(normalize_action)
            energy_cost = 0.0 if energy_cost < 0.01*self.energy_cost_weight else energy_cost
            reward = reward - energy_cost
            # print("energy cost: {}".format(energy_cost))

            # we also punish large roll & pitch angle
            pitch_angle = observations[3]
            roll_angle = observations[4]
            attitude_cost = self.attitude_cost_weight*numpy.linalg.norm(numpy.array([pitch_angle, roll_angle]))
            attitude_cost = 0.0 if attitude_cost < 0.01*self.attitude_cost_weight else attitude_cost
            reward = reward - attitude_cost
            # print("attitude cost: {}".format(attitude_cost))

        else:
            
            if self.is_in_desired_position(current_position, self.desired_point_epsilon):
                reward = self.done_reward
            else:
                reward = -1*self.done_reward


        self.previous_distance_from_des_point = distance_from_des_point


        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward


    # Internal TaskEnv Methods
    
    def is_in_desired_position(self,current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """
        
        is_in_desired_pos = False
        
        
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        z_pos_plus = self.desired_point.z + epsilon
        z_pos_minus = self.desired_point.z - epsilon
        
        x_current = current_position.x
        y_current = current_position.y
        z_current = current_position.z
        
        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        z_pos_are_close = (z_current <= z_pos_plus) and (z_current > z_pos_minus)
        
        is_in_desired_pos = x_pos_are_close and y_pos_are_close and z_pos_are_close
        
        rospy.logdebug("###### IS DESIRED POS ? ######")
        rospy.logdebug("current_position"+str(current_position))
        rospy.logdebug("x_pos_plus"+str(x_pos_plus)+",x_pos_minus="+str(x_pos_minus))
        rospy.logdebug("y_pos_plus"+str(y_pos_plus)+",y_pos_minus="+str(y_pos_minus))
        rospy.logdebug("z_pos_plus"+str(z_pos_plus)+",z_pos_minus="+str(z_pos_minus))
        rospy.logdebug("x_pos_are_close"+str(x_pos_are_close))
        rospy.logdebug("y_pos_are_close"+str(y_pos_are_close))
        rospy.logdebug("z_pos_are_close"+str(z_pos_are_close))
        rospy.logdebug("is_in_desired_pos"+str(is_in_desired_pos))
        rospy.logdebug("############")
        
        return is_in_desired_pos
    
    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)
    
        return distance
    
    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))
    
        distance = numpy.linalg.norm(a - b)
    
        return distance
        
    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]
    
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw
        
    def is_inside_workspace(self,current_position):
        """
        Check if the pelican is inside the Workspace defined
        """
        is_inside = False

        # rospy.logwarn("##### INSIDE WORK SPACE? #######")
        # rospy.logwarn("XYZ current_position"+str(current_position))
        # rospy.logwarn("work_space_x_max"+str(self.work_space_x_max)+",work_space_x_min="+str(self.work_space_x_min))
        # rospy.logwarn("work_space_y_max"+str(self.work_space_y_max)+",work_space_y_min="+str(self.work_space_y_min))
        # rospy.logwarn("work_space_y_max"+str(self.work_space_z_max)+",work_space_y_min="+str(self.work_space_z_min))
        # rospy.logwarn("############")

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
                    is_inside = True
        
        return is_inside
        
    

