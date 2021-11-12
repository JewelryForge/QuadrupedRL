import pybullet as p
import time
import pybullet_data
from utils import analyse_joint_info

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
# agentId = p.loadURDF("/home/jewel/Workspaces/teacher-student/urdf/aliengo/xacro/aliengo.urdf", startPos,
#                      startOrientation)

agentId = p.loadURDF("/home/jewel/Workspaces/teacher-student/urdf/a1/a1.urdf", startPos, startOrientation)

# print(p.getNumJoints(agentId))

p.createConstraint(
    parentBodyUniqueId=agentId,
    parentLinkIndex=-1,
    childBodyUniqueId=-1,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 1],
    childFrameOrientation=[0, 0, 0, 0.1])

for i in range(p.getNumJoints(agentId)):
    # print(p.getJointInfo(agentId, i))
    print(analyse_joint_info(p.getJointInfo(agentId, i)))

for i in range(10000):
    p.setJointMotorControl2(agentId, 8, p.TORQUE_CONTROL, force=40)  # 109微动 110风车
    # p.setJointMotorControl2(agentId, 8, p.VELOCITY_CONTROL, targetVelocity=-0.1)
    p.stepSimulation()
    time.sleep(1. / 240.)

p.disconnect()

while True:
    p.stepSimulation()
