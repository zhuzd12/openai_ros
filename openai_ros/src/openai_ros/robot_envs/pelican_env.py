import numpy
import rospy
import time
from openai_ros import robot_gazebo_env
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from mav_msgs.msg import Actuators

class PelicanEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all WamvEnv environments.
    """

    def __init__(self):
        """
        Initializes a new PelicanEnv environment.
        
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that th stream of data doesn't flow. This is for simulations
        that are pause for whatever the reason
        2) If the simulation was running already for some reason, we need to reset the controlers.
        This has to do with the fact that some plugins with tf, dont understand the reset of the simulation
        and need to be reseted to work properly.
        
        The Sensors: The sensors accesible are the ones considered usefull for AI learning.
        
        Sensor Topic List:
        * /pelican/odometry_sensor1/odometry: Odometry of the Base of Wamv
        
        Actuators Topic List: 
        * /pelican/command/motor_speed: You publish the speed of the left and right propellers.
        
        Args:
        """
        rospy.logdebug("Start PelicanEnv INIT...")
        # Variables that we give through the constructor.
        # None in this case

        # Internal Vars
        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(PelicanEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False,
                                            reset_world_or_sim="NO_RESET_SIM")


        #pos_data = None
        #takeoff = True
        #while pos_data is None or takeoff is False:
            #pos_data = rospy.wait_for_message('/pelican/ground_truth/pose', Pose, timeout=10)
            #pos_err = np.array([pos_data.position.x-0, pos_data.position.y-0, pos_data.position.z-1])
            #pos_err = math.sqrt(sum(i * i for i in pos_err))
            #if (pos_err < 0.2):
                #takeoff = True
            #else:
                #pass
        #print("take off successfully")
        rospy.logerr("PelicanEnv unpause1...")
        self.gazebo.unpauseSim()
        #self.controllers_object.reset_controllers()
        
        self._check_all_systems_ready()


        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/pelican/odometry_sensor1/odometry", Odometry, self._odom_callback)
        

        self.publishers_array = []
        self._cmd_drive_pub = rospy.Publisher('/pelican/command/motor_speed', Actuators, queue_size=1)
        
        self.publishers_array.append(self._cmd_drive_pub)

        self._check_all_publishers_ready()

        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished PelicanEnv INIT...")

    # Methods needed by the RobotGazeboEnv
    # ----------------------------
    

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        rospy.logdebug("PelicanEnv check_all_systems_ready...")
        self._check_all_sensors_ready()
        rospy.logdebug("END PelicanEnv _check_all_systems_ready...")
        return True


    # CubeSingleDiskEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        self._check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")

        
    def _check_odom_ready(self):
        self.odom = None
        rospy.logdebug("Waiting for /pelican/odometry_sensor1/odometry to be READY...")
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/pelican/odometry_sensor1/odometry", Odometry, timeout=1.0)
                rospy.logdebug("Current /pelican/odometry_sensor1/odometry READY=>")

            except:
                rospy.logerr("Current /pelican/odometry_sensor1/odometry not ready yet, retrying for getting odom")
        return self.odom
        
        
    
    def _odom_callback(self, data):
        self.odom = data
    
    
    def _check_all_publishers_ready(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rospy.logdebug("START ALL SENSORS READY")
        for publisher_object in self.publishers_array:
            self._check_pub_connection(publisher_object)
        rospy.logdebug("ALL SENSORS READY")

    def _check_pub_connection(self, publisher_object):

        rate = rospy.Rate(10)  # 10hz
        while publisher_object.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to publisher_object yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("publisher_object Publisher Connected")

        rospy.logdebug("All Publishers READY")
        
    
    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    
    def wait_time_for_execute_movement(self, time_sleep):
        """
        Because this Wamv position is global, we really dont have
        a way to know if its moving in the direction desired, because it would need
        to evaluate the difference in position and speed on the local reference.
        """
        time.sleep(time_sleep)
    
    def get_odom(self):
        return self.odom