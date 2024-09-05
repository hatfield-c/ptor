import numpy as np
import pybullet as pb
import time

import physics.Transform as Transform
import sensors.SensorInterface as SensorInterface

class DepthSensor(SensorInterface.SensorInterface):
	def __init__(
			self,
			local_position,
		):
		self.local_position = local_position
		self.offset_distance = np.linalg.norm(self.offset)

		self.avg_render_time = 0
		self.current_time = time.time()

		magnitude = self.offset_distance
		if magnitude == 0:
			magnitude = 1

		self.offset_direction = self.offset / magnitude

	def ReadSensor(self, control_data):
		
		time_diff = time.time() - self.current_time
		self.current_time = time.time()
		self.avg_render_time = (self.avg_render_time + time_diff) / 2

		return 0
