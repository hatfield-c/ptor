import torch

import entity.EntityInterface as EntityInterface
import entity.actuator.RotorActuator as RotorActuator
import entity.sensor.DepthSensor as DepthSensor
import engine.Rigidbody as Rigidbody
import engine.Quaternion as Quaternion
import engine.Transform as Transform

import ai.planner.PidAlignmentPlanner as PidAlignmentPlanner
import ai.controller.PidForwardController as PidForwardController

class DroneTau(EntityInterface.EntityInterface):
	def __init__(self):
		self.camera_position = torch.FloatTensor([[0, 0.209219, 0.031208]]).cuda()
		
		self.rigidbody = Rigidbody.Rigidbody()
		
		self.planner = PidAlignmentPlanner.PidAlignmentPlanner()
		self.controller = PidForwardController.PidForwardController()
		self.rotor_actuator = RotorActuator.RotorActuator(self.rigidbody)
		self.camera_sensor = DepthSensor.DepthSensor()
		
		self.desired_direction = torch.FloatTensor([[0, 1, 0]]).cuda()
	
	def Update(self):
		current_state = {
			"position": self.rigidbody.body_origin[0],
			"velocity": self.rigidbody.body_velocity[0],
			"angular_velocity": self.rigidbody.body_angular_velocity[0],
			"quaternion": self.rigidbody.body_rotation,
		}
		
		plan = self.planner.GetPlan(current_state)
		
		if plan is None:
			return
		
		plan["rigidbody"] = self.rigidbody
		control_signals = self.controller.GetControlSignal(plan)
		self.rotor_actuator.Actuate(control_signals)
			
		self.desired_direction = plan["desired_direction"].view(1, -1) * 0.5
		
	def GetCameraPosition(self):
		camera_offset = Quaternion.RotatePoints(self.camera_position, self.rigidbody.body_rotation)
		camera_position = self.rigidbody.body_origin + camera_offset
		
		return camera_position