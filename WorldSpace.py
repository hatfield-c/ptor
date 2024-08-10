import torch

import CONFIG
import Quaternion

class WorldSpace:
	def __init__(self):
		# dimensions: 100m x 100m x 30m
		# granularity: 0.1m x 0.1m x 0.1m
		# array_size: 1000 x 1000 x 300
		self.space = torch.zeros(1000, 1000, 300).cuda()
		
		self.indices_per_meter = 10
		
		self.LoadWorldVoxels()
		
	def LoadWorldVoxels(self):
		particle_data = torch.load(CONFIG.static_particles_path).cuda()
		particle_positions = particle_data[:, :3]
		material_vals = particle_data[:, 3]
		
		world_indices = (particle_positions * self.indices_per_meter).int()
		world_indices = torch.clamp(
			world_indices, 
			torch.zeros((1, 3)).cuda().int(), 
			torch.IntTensor(list(self.space.shape)).cuda().view(1, 3) - 1
		)

		self.space[world_indices[:, 0], world_indices[:, 1], world_indices[:, 2]] = material_vals
		
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
		render_space[particle_indices[:, 0], particle_indices[:, 1], particle_indices[:, 2]] = rigidbody.material_val
		
		return render_space
		