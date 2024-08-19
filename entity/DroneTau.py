import torch

import entity.EntityInterface as EntityInterface
import entity.actuator.RotorActuator as RotorActuator
import engine.Rigidbody as Rigidbody
import engine.Quaternion as Quaternion

import ai.planner.PidAlignmentPlanner as PidAlignmentPlanner
import ai.controller.PidForwardController as PidForwardController

class DroneTau(EntityInterface.EntityInterface):
	def __init__(self):
		self.rigidbody = Rigidbody.Rigidbody()
		
		self.planner = PidAlignmentPlanner.PidAlignmentPlanner()
		self.controller = PidForwardController.PidForwardController()
		self.rotor_actuator = RotorActuator.RotorActuator(self.rigidbody)
		
		self.desired_direction = torch.FloatTensor([[0, 1, 0]]).cuda()
	
	def Update(self):
		current_state = {
			"position": self.rigidbody.body_origin[0],
			"velocity": self.rigidbody.body_velocity[0],
			"angular_velocity": self.rigidbody.body_angular_velocity[0],
			"quaternion": self.rigidbody.body_rotation,
		}
		
		plan = self.planner.GetPlan(current_state)
		plan["rigidbody"] = self.rigidbody
		control_signals = self.controller.GetControlSignal(plan)
		self.rotor_actuator.Actuate(control_signals)
			
		self.desired_direction = plan["desired_direction"].view(1, -1) * 0.5