import torch

import CONFIG
import Quaternion

class WorldSpace:
	def __init__(self):
		# dimensions: 100m x 100m x 30m
		# granularity: 0.1m x 0.1m x 0.1m
		# array_size: 1000 x 1000 x 300
		self.space = torch.zeros(1000, 1000, 300).cuda()
		#self.space[:, :, 0:10] = 1
		#self.space[430:480, 430:480, 10:80] = 1
		
		self.indices_per_meter = 10
		
		self.BuildCubicMeters(
			torch.FloatTensor([0, 0, 0]), 
			torch.FloatTensor([100, 100, 1]), 
			1
		)
		
		self.BuildCubicMeters(
			torch.FloatTensor([43, 43, 1]), 
			torch.FloatTensor([48, 48, 8]), 
			1
		)
		
	def BuildCubicMeters(self, start, end, value):
		start = (start * self.indices_per_meter).int()
		end = (end * self.indices_per_meter).int()

		self.space[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = value
		
	def GetParticleData(self):
		return self.particle_positions, self.particle_sizes
	
	def GetRenderSpace(self, rigidbody):
		particle_positions, particle_sizes = rigidbody.GetParticleData()

		particle_indices = (particle_positions * self.indices_per_meter)
		particle_indices = particle_indices.int()
		
		render_space = self.space.clone()
		render_space[particle_indices[:, 0], particle_indices[:, 1], particle_indices[:, 2]] = 2
		
		return render_space
		