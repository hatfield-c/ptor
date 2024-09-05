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
			p_scale = 0.3,
			i_scale = 0,
			d_scale = 0.5,
		)
		
		self.goals = {
			"max_altitude": 4
		}
		
		self.tasks = [
			#"takeoff",
			#"point up",
			"linear_path",
			#"landing"
		]
		
		self.task_index = 0
		self.task_time_start = time.time()
		self.max_task_time = {
			"takeoff": 5,
			"point_up": 10,
			"linear_path": 60,
			"landing": 10
		}

	def GetPlan(self, current_state):
		
		# todo: encapsulate in try catch, with land behavior always being re-added
		
		
		task = self.GetTask(current_state, self.task_index)
		
		if task == "takeoff":
			pass
		elif task == "point_up":
			pass
		elif task == "linear_path":
			plan = self.LinearPath(current_state)
		elif task == "landing" or "force_land":
			plan = self.LandWhileBraking(current_state)
		
		return plan

	def GetTask(self, current_state, task_index):
		
		task = self.tasks[self.task_index]
		
		if task == "takeoff":
			#altitude = current_state["altitude"]
			pass
		
		if task == "point_up":
			pass
		
		if task == "linear_path":			
			position = current_state["position"]
			
			if self.IsTaskTooLong(task) or position[1] > 30:
				task = self.NextTask()
				
		if task == "landing" or "force_land":
			task = task
		
		return task

	def NextTask(self):
		self.task_index += 1
		self.task_index = min(self.task_index, len(self.tasks) - 1)
		
		self.task_time_start = time.time()
		
		return self.tasks[self.task_index]
	
	def IsTaskTooLong(self, task):
		max_time = self.max_task_time[task]
		elapsed_time = time.time() - self.task_time_start
		
		return elapsed_time > max_time

	def LandWhileBraking(self, current_state):
		current_position = current_state["position"]
		velocity = current_state["velocity"]
		current_quat = current_state["quaternion"]
		
		speed = torch.linalg.norm(velocity)
		
		#if speed 

	def LinearPath(self, current_state):
		current_position = current_state["position"]
		velocity = current_state["velocity"]
		current_quat = current_state["quaternion"]
		
		lateral_error = current_position[0] - 50
		lateral_signal = self.align_pid.ControlStep(lateral_error, 0, velocity[0])
		
		lateral_signal = torch.clip(lateral_signal, -1, 1)
		forward_signal = torch.sqrt(1 - (lateral_signal ** 2))
		
		desired_direction = torch.FloatTensor([lateral_signal, forward_signal]).cuda()

		desired_altitude = 6.5

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