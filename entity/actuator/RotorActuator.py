import numpy as np

import actuators.ActuatorInterface as ActuatorInterface

class RotorActuator(ActuatorInterface.ActuatorInterface):
	def __init__(self, rotor_max = 0.25, torque_max = 0.5):
		self.last_command = None
		self.rotor_max = rotor_max
		self.torque_max = torque_max
		self.client_id = client_id

	def Actuate(self, control_data):

		fr_rotor = control_data["fr_rotor_force"]
		fl_rotor = control_data["fl_rotor_force"]
		br_rotor = control_data["br_rotor_force"]
		bl_rotor = control_data["bl_rotor_force"]
		torque = control_data["torque"]

		#if "thrust_signal" in control_data:
		#	self.last_command = np.array([
		#		control_data["thrust_signal"],
		#		control_data["pitch_signal"],
		#		control_data["roll_signal"],
		#		control_data["yaw_signal"],
		#	])
		#else:
		#	self.last_command = np.zeros(4)

		#self.last_command = np.array([
		#	fr_rotor,
		#	fl_rotor,
		#	br_rotor,
		#	bl_rotor,
		#	torque
		#])

		fr_rotor = np.clip(fr_rotor, 0, self.rotor_max)
		fl_rotor = np.clip(fl_rotor, 0, self.rotor_max)
		br_rotor = np.clip(br_rotor, 0, self.rotor_max)
		bl_rotor = np.clip(bl_rotor, 0, self.rotor_max)
		torque = np.clip(torque, -self.torque_max, self.torque_max)

		#desired_direction = control_data["desired_direction"]
		#self.last_command = np.array([
		#	desired_direction[0],
		#	desired_direction[1],
		#	control_data["desired_altitude"]
		#])

		pb.applyExternalForce(
			control_data["pb_id"],
			0,
			forceObj = [0, 0, fr_rotor],
			posObj = [0, 0, 0],
			flags = pb.LINK_FRAME,
			physicsClientId = self.client_id
		)
		pb.applyExternalForce(
			control_data["pb_id"],
			1,
			forceObj = [0, 0, fl_rotor],
			posObj = [0, 0, 0],
			flags = pb.LINK_FRAME,
			physicsClientId = self.client_id
		)
		pb.applyExternalForce(
			control_data["pb_id"],
			2,
			forceObj = [0, 0, br_rotor],
			posObj = [0, 0, 0],
			flags = pb.LINK_FRAME,
			physicsClientId = self.client_id
		)
		pb.applyExternalForce(
			control_data["pb_id"],
			3,
			forceObj = [0, 0, bl_rotor],
			posObj = [0, 0, 0],
			flags = pb.LINK_FRAME,
			physicsClientId = self.client_id
		)

		pb.applyExternalTorque(
			control_data["pb_id"],
			-1,
			torqueObj = [0, 0, torque],
			flags = pb.LINK_FRAME,
			physicsClientId = self.client_id
		)

	def GetLastCommand(self):
		return self.last_command
