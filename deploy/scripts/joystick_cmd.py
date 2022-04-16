#!/usr/bin/env python

import math
import os
import time

import rospy
from alienGo_deploy.msg import FloatArray


class JoystickRosInterface(object):
    def __init__(self, topic_name='/cmd_vel', gamepad_type='PS4'):
        from gamepad import gamepad, controllers
        if not gamepad.available():
            print('Please connect your gamepad...')
            while not gamepad.available():
                time.sleep(1.0)
        try:
            self.gamepad = getattr(controllers, gamepad_type)()
        except AttributeError:
            raise RuntimeError('`{}` is not supported, all {}'.format(gamepad_type, controllers.all_controllers))
        self.cmd_pub = rospy.Publisher(topic_name, FloatArray, queue_size=1)
        self.gamepad.startBackgroundUpdates()
        print('Gamepad connected')

    @staticmethod
    def is_available():
        from gamepad import gamepad
        return gamepad.available()

    def get_cmd_and_publish(self):
        if self.gamepad.isConnected():
            try:
                x_speed = -self.gamepad.axis('LEFT-Y')
                y_speed = -self.gamepad.axis('LEFT-X')
                steering = -self.gamepad.axis('RIGHT-X')
                steering = 1. if steering > 0.2 else -1. if steering < -0.2 else 0.
                speed_norm = math.hypot(x_speed, y_speed)
                msg = FloatArray()
                if speed_norm:
                    msg.data = [x_speed / speed_norm, y_speed / speed_norm, steering]
                else:
                    msg.data = [0., 0., steering]
                self.cmd_pub.publish(msg)
            except Exception as e:
                print(e)
                os._exit(1)
        else:
            print('Gamepad Disconnected')
            os._exit(0)

    def __del__(self):
        self.gamepad.disconnect()


if __name__ == '__main__':
    rospy.init_node('joystick_ros_interface')
    joystick = JoystickRosInterface()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        joystick.get_cmd_and_publish()
        rate.sleep()
    os._exit(0)
