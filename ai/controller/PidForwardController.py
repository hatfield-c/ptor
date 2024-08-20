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
		self.move_angle = -math.pi / 12
		self.move_depth = math.sin(self.move_angle)
		self.move_xy_norm = math.cos((math.pi / 2) + self.move_angle)
		self.move_z_target = math.sin((math.pi / 2) + self.move_angle)
		self.move_z_target = torch.FloatTensor([[self.move_z_target]]).cuda()

		self.thrust_pid = Pid.Pid(
			p_scale = 1,
			i_scale = 0,
			d_scale = 2,
			#debug = True
		)
		self.pitch_pid = Pid.Pid(
			p_scale = 1,
			i_scale = 0,
			d_scale = 2,
			#debug = True
		)
		self.roll_pid = Pid.Pid(
			p_scale = 1,
			i_scale = 0,
			d_scale = 2,
			#debug = True
		)
		self.yaw_pid = Pid.Pid(
			p_scale = 0.5,
			i_scale = 0,
			d_scale = 2,
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

		inverse_quat = Quaternion.GetQuaternionInverse(current_quat)

		#desired_direction = torch.FloatTensor([[1, 0]]).cuda()
		desired_direction = desired_direction.view(1, -1)
		
		desired_unit = Transform.GetUnit(desired_direction)
		move_xy_dir = desired_unit * self.move_xy_norm				
		desired_local = torch.cat((move_xy_dir, self.move_z_target), dim = 1)
		desired_local = Quaternion.RotatePoints(desired_local, inverse_quat)
		axis_strengths = desired_local[:, [0, 1]]
		
		body_forward = Transform.GetForward(current_quat)
		forward_xy = body_forward[[0], [0, 1]]
		forward_xy = Transform.GetUnit(forward_xy)
		
		desired_shifted  = desired_unit[0, [1, 0]]
		desired_shifted[0] = -desired_shifted[0]
		desired_shifted = desired_shifted.view(1, -1)
		desired_alignment = torch.sum(forward_xy * desired_shifted)
		
		pitch_signal = axis_strengths[0, 1]
		roll_signal = axis_strengths[0, 0]
		yaw_signal = -desired_alignment
		
		thrust_rpm = self.thrust_pid.ControlStep(current_altitude, desired_altitude, velocity[2])
		pitch_rpm = self.pitch_pid.ControlStep(pitch_signal)
		roll_rpm = self.roll_pid.ControlStep(roll_signal)
		yaw_rpm = self.yaw_pid.ControlStep(yaw_signal)
		
		#print(roll_error, roll_rpm)
		#print(yaw_error, yaw_rpm)
		#print("")
		
		control_data = self.MotorMixer(thrust_rpm, pitch_rpm, roll_rpm, yaw_rpm)

		control_data["drop_package"] = plan["drop_package"]
		
		return control_data

	def MotorMixer(self, thrust, pitch, roll, yaw):
		motor_vals = {}

		thrust = torch.clip(thrust, 0, 1)
		pitch = torch.clip(pitch, -1, 1)
		roll = torch.clip(roll, -1, 1)
		yaw = torch.clip(yaw, -1, 1)
		angle_strength = torch.abs(pitch) + torch.abs(roll) + torch.abs(yaw)
		pitch_strength = torch.abs(pitch)
		roll_strength = torch.abs(roll)
		yaw_strength = torch.abs(yaw)
		
		thrust_share = max(0.7, 1 - angle_strength)
		pitch_share = 0
		roll_share = 0
		yaw_share = 0

		if angle_strength > 0:
			angle_share = 1 - thrust_share

			pitch_share = angle_share * (pitch_strength / angle_strength)
			roll_share = angle_share * (roll_strength / angle_strength)
			yaw_share = angle_share * (yaw_strength / angle_strength)
			
		thrust = min(thrust, thrust_share)
		pitch = min(pitch_strength, pitch_share) * torch.sign(pitch)
		roll = min(roll_strength, roll_share) * torch.sign(roll)
		yaw = min(yaw_strength, yaw_share) * torch.sign(yaw)
		
		fr = thrust + pitch + roll + yaw
		fl = thrust + pitch - roll - yaw
		br = thrust - pitch + roll - yaw
		bl = thrust - pitch - roll + yaw
		
		min_throttle = min(0, fr)
		min_throttle = min(min_throttle, fl)
		min_throttle = min(min_throttle, br)
		min_throttle = min(min_throttle, bl)
		throttle_offset = abs(min_throttle)
		
		fr += throttle_offset
		fl += throttle_offset
		br += throttle_offset
		bl += throttle_offset
		
		max_throttle = max(0, fr)
		max_throttle = max(max_throttle, fl)
		max_throttle = max(max_throttle, br)
		max_throttle = max(max_throttle, bl)
		
		if max_throttle > 1:
			fr = fr / max_throttle
			fl = fl / max_throttle
			br = br / max_throttle
			bl = bl / max_throttle
		
		#print("{:.4f}".format(fr), "{:.4f}".format(fl), "{:.4f}".format(br), "{:.4f}".format(bl))
		#print("    {:.4f}".format(thrust), "{:.4f}".format(pitch), "{:.4f}".format(roll), "{:.4f}".format(yaw))
		#input()

		motor_vals["fr_throttle"] = fr
		motor_vals["fl_throttle"] = fl
		motor_vals["br_throttle"] = br
		motor_vals["bl_throttle"] = bl

		return motor_vals
