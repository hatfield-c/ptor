import torch
import math

import entity.actuator.ActuatorInterface as ActuatorInterface
import engine.Quaternion as Quaternion

class RotorActuator(ActuatorInterface.ActuatorInterface):
	def __init__(self, rigidbody, rotor_max = 0.5, torque_max = 0.5):
		self.rigidbody = rigidbody
		self.rotor_max = rotor_max
		self.torque_max = torque_max
		self.last_command = None
		
		self.flag = True
		
		self.motor_positions = torch.FloatTensor([
			[0.144216, 0.107853, 0.03868],
			[-0.144216, 0.107853, 0.03868],
			[0.144216, -0.107853, 0.03868],
			[-0.144216, -0.107853, 0.03868],
		]).cuda()

	def Actuate(self, control_data):

		fr_rotor = control_data["fr_rotor_force"]
		fl_rotor = control_data["fl_rotor_force"]
		br_rotor = control_data["br_rotor_force"]
		bl_rotor = control_data["bl_rotor_force"]
		torque = control_data["torque"]

		fr_rotor = torch.clip(fr_rotor, 0, self.rotor_max)
		fl_rotor = torch.clip(fl_rotor, 0, self.rotor_max)
		br_rotor = torch.clip(br_rotor, 0, self.rotor_max)
		bl_rotor = torch.clip(bl_rotor, 0, self.rotor_max)
		torque = torch.clip(torque, -self.torque_max, self.torque_max)
		
		self.ActuateMotor(self.motor_positions[[0]], fr_rotor)
		self.ActuateMotor(self.motor_positions[[1]], fl_rotor)
		self.ActuateMotor(self.motor_positions[[2]], br_rotor)
		self.ActuateMotor(self.motor_positions[[3]], bl_rotor)

		torque = torch.FloatTensor([[0, 0, torque]]).cuda()
		torque = Quaternion.RotatePoints(torque, self.rigidbody.body_rotation)
		self.rigidbody.AddTorque(torque.T)
		
		#pb.applyExternalTorque(
		#	control_data["pb_id"],
		#	-1,
		#	torqueObj = [0, 0, torque],
		#	flags = pb.LINK_FRAME,
		#	physicsClientId = self.client_id
		#)

	def ActuateMotor(self, motor_position, thrust):
		force = torch.FloatTensor([[0, 0, thrust]]).cuda()
		
		force = Quaternion.RotatePoints(force, self.rigidbody.body_rotation)
		motor_position = Quaternion.RotatePoints(motor_position, self.rigidbody.body_rotation)
		
		self.rigidbody.AddForce(force, motor_position)		

	def GetLastCommand(self):
		return self.last_command
