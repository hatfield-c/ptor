import time
import math
import random
import torch

import ai.controller.modules.Pid as Pid
import engine.Transform as Transform
import ai.planner.PlannerInterface as PlannerInterface

class PidAlignmentPlanner(PlannerInterface.PlannerInterface):

	def __init__(self):
		self.align_pid = Pid.Pid(
			p_scale = 0.5,
			i_scale = 0,
			d_scale = 1,
		)

	def GetPlan(self, current_state):
		current_position = current_state["position"]
		velocity = current_state["velocity"]
		current_quat = current_state["quaternion"]
		
		lateral_error = current_position[0] - 50
		lateral_signal = self.align_pid.ControlStep(lateral_error, 0, velocity[0])
		
		lateral_signal = torch.clip(lateral_signal, -1, 1)
		forward_signal = torch.sqrt(1 - (lateral_signal ** 2))
		
		desired_direction = torch.FloatTensor([lateral_signal, forward_signal]).cuda()

		desired_altitude = 15

		drop_package = False

		plan = {
			"action": "align",
			"current_quat": current_quat,
			"current_altitude": current_position[2],
			"desired_direction": desired_direction[[0, 1]],
			"desired_altitude": desired_altitude,
			"velocity": velocity,
			"drop_package": drop_package
		}

		return plan

	def SetNewPath(self, control_points):
		pass

	def GetControlPoints(self):
		control_points = torch.zeros((4, 1)).cuda()

		return control_points
