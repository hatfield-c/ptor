import torch

import Quaternion

class Rigidbody:
	def __init__(self):
		x = torch.linspace(0, 1, 3)
		y = torch.linspace(0, 1, 3)
		z = torch.linspace(0, 1, 3)
		
		position_mesh = torch.meshgrid([x, y, z], indexing = "xy")
		position_mesh = torch.stack(position_mesh)
		position_mesh = position_mesh.reshape(3, -1).T
		
		self.particle_positions = position_mesh.cuda()
		self.alpha_positions = position_mesh.clone()
		
		self.particle_count = self.particle_positions.shape[0]
		self.particle_dimensionality = self.particle_positions.shape[1]
		
		self.particle_sizes = torch.zeros(self.particle_count).cuda() + 0.5
		self.particle_masses = torch.ones(self.particle_count).cuda()
		
		self.body_velocity = 0
		self.body_mass = torch.sum(self.particle_masses)
		
		weighted_positions = self.particle_positions * self.particle_masses.reshape((self.particle_count, 1))
		self.body_origin = torch.sum(weighted_positions)
		
		self.body_rotation = torch.FloatTensor([0, 0, 0, 1])
		
		self.ambient_color = torch.FloatTensor([255, 0, 0])
		self.shiny_color = torch.FloatTensor([255, 255, 255])
		self.shiny_val = 1
		
	def Update(self):
		self.particle_positions = Quaternion.RotatePoints(self.alpha_positions, self.body_rotation)
		
	def GetParticleData(self):
		return self.particle_positions, self.particle_sizes