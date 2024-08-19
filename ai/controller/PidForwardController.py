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
		self.move_angle = math.pi / 6
		self.move_depth = -math.sin(self.move_angle)

		self.thrust_pid = Pid.Pid(
			p_scale = 0.1,
			i_scale = 0,
			d_scale = 0,
			#debug = True
		)
		self.pitch_pid = Pid.Pid(
			p_scale = 0.1,
			i_scale = 0,
			d_scale = 1,
			#debug = True
		)
		self.roll_pid = Pid.Pid(
			p_scale = 0.1,
			i_scale = 0,
			d_scale = 2,
			#debug = True
		)
		self.yaw_pid = Pid.Pid(
			p_scale = 0.4,
			i_scale = 0,
			d_scale = 5,
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
		yaw_error = local_corner_1[0, 2] - local_corner_2[0, 2]

		thrust_rpm = self.thrust_pid.ControlStep(current_altitude, desired_altitude, velocity[2])
		pitch_rpm = self.pitch_pid.ControlStep(pitch_error)
		roll_rpm = self.roll_pid.ControlStep(roll_error)
		yaw_rpm = self.yaw_pid.ControlStep(yaw_error)
		
		#print(roll_error, roll_rpm)
		#print(yaw_error, yaw_rpm)
		#print("")

		control_data = self.MotorMixer(thrust_rpm, yaw_rpm, pitch_rpm, roll_rpm)

		control_data["drop_package"] = plan["drop_package"]
		
		return control_data

	def MotorMixer(self, thrust, yaw, pitch, roll):
		motor_vals = {}

		thrust = max(thrust, 0)
		thrust = min(thrust, 0.1)
		

		normalizer = torch.abs(thrust) + torch.abs(yaw) + torch.abs(pitch) + torch.abs(roll) 

		fr = (thrust + yaw + pitch + roll) / normalizer
		fl = (thrust - yaw + pitch - roll) / normalizer
		br = (thrust - yaw - pitch + roll) / normalizer
		bl = (thrust + yaw - pitch - roll) / normalizer
		
		#print("{:.2f}".format(fr), "{:.2f}".format(fl), "{:.2f}".format(br), "{:.2f}".format(bl))
		#print("    {:.2f}".format(thrust), "{:.2f}".format(pitch), "{:.2f}".format(pitch), "{:.2f}".format(roll))
		#input()

		motor_vals["fr_throttle"] = fr
		motor_vals["fl_throttle"] = fl
		motor_vals["br_throttle"] = br
		motor_vals["bl_throttle"] = bl

		return motor_vals
