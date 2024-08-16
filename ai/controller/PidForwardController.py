import math
import torch

import CONFIG
import engine.Quaternion as Quaternion
import engine.Transform as Transform
import ai.controller.ControllerInterface as ControllerInterface
import ai.controller.modules.Pid as Pid

class PidForwardController(ControllerInterface.ControllerInterface):
	def __init__(self):

		self.xy_corner_1 = torch.FloatTensor([[1, -1, 0]]).cuda()
		self.xy_corner_2 = torch.FloatTensor([[-1, -1, 0]]).cuda()

		self.speed_val = 1
		self.move_angle = math.pi / 8
		self.move_depth = -math.sin(self.move_angle)

		self.thrust_pid = Pid.Pid(
			p_scale = 0.05,
			i_scale = 0,
			d_scale = 0.5,
			#debug = True
		)
		self.pitch_pid = Pid.Pid(
			p_scale = 0.01,
			i_scale = 0,
			d_scale = 0.05,
			#debug = True
		)
		self.roll_pid = Pid.Pid(
			p_scale = 0.001,
			i_scale = 0,
			d_scale = 0.05,
			#debug = True
		)
		self.yaw_pid = Pid.Pid(
			p_scale = 0.001,
			i_scale = 0,
			d_scale = 0.05,
			#debug = True
		)

	def GetControlSignal(self, plan):
		control_signal = {}

		action = plan["action"]

		if action == "align":
			control_signal = self.AlignAction(plan)

		return control_signal


	def AlignAction(self, plan):
		current_altitude = plan["current_altitude"]
		desired_direction = plan["desired_direction"]
		desired_altitude = plan["desired_altitude"]
		velocity = plan["velocity"]
		current_quat = plan["current_quat"]

		desired_xy = desired_direction[[0, 1]]
		unitFactor = torch.linalg.norm(desired_xy)
		if unitFactor == 0:
			unitFactor = 1
		desired_xy = desired_xy / unitFactor

		local_front = Transform.GetForward(current_quat)
		local_corner_1 = Quaternion.RotatePoints(self.xy_corner_1, current_quat)
		local_corner_2 = Quaternion.RotatePoints(self.xy_corner_2, current_quat)

		local_corner_1_xy = local_corner_1[[0], [0, 1]]
		local_corner_2_xy = local_corner_2[[0], [0, 1]]
		local_corner_1_xy = local_corner_1_xy / torch.linalg.norm(local_corner_1_xy)
		local_corner_2_xy = local_corner_2_xy / torch.linalg.norm(local_corner_2_xy)

		dist_1_xy = desired_xy - local_corner_1_xy
		dist_2_xy = desired_xy - local_corner_2_xy
		dist_1_xy = torch.linalg.norm(dist_1_xy)
		dist_2_xy = torch.linalg.norm(dist_2_xy)

		pitch_error = local_front[0, 2] - self.move_depth
		roll_error = dist_2_xy - dist_1_xy
		yaw_error = local_corner_2[0, 2] - local_corner_1[0, 2]

		thrust_rpm = self.thrust_pid.ControlStep(current_altitude, desired_altitude, velocity[2])
		pitch_rpm = self.pitch_pid.ControlStep(pitch_error)
		roll_rpm = self.roll_pid.ControlStep(roll_error)
		yaw_rpm = self.yaw_pid.ControlStep(yaw_error)
		
		rigidbody = plan["rigidbody"]
		#print(rigidbody.body_velocity)
		#print(rigidbody.body_angular_velocity)
		#print(self.roll_pid.memory["estimated_velocity"])
		#print(self.yaw_pid.memory["estimated_velocity"])
		#print("======")
		#print(local_front)
		#print(desired_direction)
		#print("{:.6f}".format(pitch_error), "{:.6f}".format(pitch_rpm))
		#print("{:.6f}".format(roll_error), "{:.6f}".format(roll_rpm))
		#print("{:.6f}".format(yaw_error), "{:.6f}".format(yaw_rpm))
		#print("=======")

		control_data = self.MotorMixer(thrust_rpm, yaw_rpm, pitch_rpm, roll_rpm)

		#print(control_data["fr_rotor_force"])
		#print(control_data["fl_rotor_force"])
		#print(control_data["br_rotor_force"])
		#print(control_data["bl_rotor_force"])
		#print("_______________")
		#input()

		#control_data["desired_direction"] = desired_direction[[0, 1]]
		#control_data["desired_altitude"] = desired_altitude
		control_data["drop_package"] = plan["drop_package"]
		#control_data["thrust_signal"] = thrust_rpm
		#control_data["pitch_signal"] = pitch_rpm
		#control_data["roll_signal"] = roll_rpm
		#control_data["yaw_signal"] = yaw_rpm

		return control_data

	def MotorMixer(self, thrust, yaw, pitch, roll):
		motor_vals = {}

		fr = thrust + yaw + pitch + roll
		fl = thrust - yaw + pitch - roll
		br = thrust - yaw - pitch + roll
		bl = thrust + yaw - pitch - roll

		#print("thrust:", thrust)
		#print(fr)
		#print(fl)
		#print(br)
		#print(bl)
		#print("==============")

		motor_vals["fr_rotor_force"] = fr
		motor_vals["fl_rotor_force"] = fl
		motor_vals["br_rotor_force"] = br
		motor_vals["bl_rotor_force"] = bl

		motor_vals["torque"] = yaw

		return motor_vals
