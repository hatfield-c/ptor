import torch

import entity.EntityInterface as EntityInterface
#import entity.actuator.RotorActuator as RotorActuator
import engine.Rigidbody as Rigidbody
import engine.Quaternion as Quaternion

class DroneTau(EntityInterface.EntityInterface):
	def __init__(self):
		self.rigidbody = Rigidbody.Rigidbody()
		
		#self.rotor_actuator = RotorActuator.RotorActuator()
	
	def Update(self):
		#plan = self.rigidbody.planner.GetPlan()
			
		for i in range(4):
			thrust = torch.FloatTensor([[0, 0, 0.28]]).cuda()
			motor_position = self.rigidbody.motor_positions[[i]]
			
			thrust = Quaternion.RotatePoints(thrust, self.rigidbody.body_rotation)
			motor_position = Quaternion.RotatePoints(motor_position, self.rigidbody.body_rotation)
			
			self.rigidbody.AddForce(thrust, motor_position)