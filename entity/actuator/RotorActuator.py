import torch
import math

import CONFIG
import entity.actuator.ActuatorInterface as ActuatorInterface
import engine.Quaternion as Quaternion

class RotorActuator(ActuatorInterface.ActuatorInterface):
	def __init__(self, rigidbody):
		self.rigidbody = rigidbody
		self.last_command = None
		
		self.max_throttle = 1
		self.max_throttle_thrust = 1.03 * torch.linalg.norm(CONFIG.gravity)
		self.torque_empirical_ratio = 0.15
		
		self.motor_positions = torch.FloatTensor([
			[0.144216, 0.107853, 0.03868],
			[-0.144216, 0.107853, 0.03868],
			[0.144216, -0.107853, 0.03868],
			[-0.144216, -0.107853, 0.03868],
		]).cuda()

	def Actuate(self, control_data):

		fr_thottle = control_data["fr_throttle"]
		fl_thottle = control_data["fl_throttle"]
		br_thottle = control_data["br_throttle"]
		bl_thottle = control_data["bl_throttle"]
		
		fr_thrust = torch.clip(fr_thottle, 0, self.max_throttle) * self.max_throttle_thrust
		fl_thrust = torch.clip(fl_thottle, 0, self.max_throttle) * self.max_throttle_thrust
		br_thrust = torch.clip(br_thottle, 0, self.max_throttle) * self.max_throttle_thrust
		bl_thrust = torch.clip(bl_thottle, 0, self.max_throttle) * self.max_throttle_thrust
		
		torque = (-fr_thrust + fl_thrust + br_thrust - bl_thrust) * self.torque_empirical_ratio
		
		self.ActuateMotor(self.motor_positions[[0]], fr_thrust)
		self.ActuateMotor(self.motor_positions[[1]], fl_thrust)
		self.ActuateMotor(self.motor_positions[[2]], br_thrust)
		self.ActuateMotor(self.motor_positions[[3]], bl_thrust)

		torque = torch.FloatTensor([[0, 0, torque]]).cuda()
		torque = Quaternion.RotatePoints(torque, self.rigidbody.body_rotation)
		self.rigidbody.AddTorque(torque.T)
		
	def ActuateMotor(self, motor_position, thrust):
		force = torch.FloatTensor([[0, 0, thrust]]).cuda()
		
		force = Quaternion.RotatePoints(force, self.rigidbody.body_rotation)
		motor_position = Quaternion.RotatePoints(motor_position, self.rigidbody.body_rotation)
		
		self.rigidbody.AddForce(force, motor_position)		

	def GetLastCommand(self):
		return self.last_command
