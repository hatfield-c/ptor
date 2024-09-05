import torch

import engine.WorldSpace as WorldSpace
import entity.sensor.DepthSensor as DepthSensor

class PositionTrainer:
	def __init__(self):
		self.learning_rate = 1e-5
		
		self.world_space = WorldSpace.WorldSpace()
		self.depth_sensor = DepthSensor.DepthSensor()
	