import torch

import CONFIG
import engine.Quaternion as Quaternion

class WorldSpace:
	def __init__(self):
		# dimensions: 100m x 100m x 30m
		# granularity: 0.1m x 0.1m x 0.1m
		# array_size: 1000 x 1000 x 300
		self.space = torch.zeros(1000, 1000, 300).cuda()
		
		self.indices_per_meter = CONFIG.indices_per_meter
		
		self.LoadWorldVoxels()
		
	def LoadWorldVoxels(self):
		particle_data = torch.load(CONFIG.static_env_particles_path).cuda()
		baked_data = torch.load(CONFIG.static_env_baked_path)

		particle_positions = particle_data[:, :3]
		material_vals = particle_data[:, 3]
		
		world_indices = (particle_positions * self.indices_per_meter).int()
		world_indices = torch.clamp(
			world_indices, 
			torch.zeros((1, 3)).cuda().int(), 
			torch.IntTensor(list(self.space.shape)).cuda().view(1, 3) - 1
		)

		self.space[world_indices[:, 0], world_indices[:, 1], world_indices[:, 2]] = material_vals
		
		self.space[500, :, 20] = 6
		self.space[320, 200, 20] = 6
		
	def BuildCubicMeters(self, start, end, value):
		start = (start * self.indices_per_meter).int()
		end = (end * self.indices_per_meter).int()

		self.space[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = value
		
	def GetParticleData(self):
		return self.particle_positions, self.particle_sizes
	
	def GetRenderSpace(self, entity):
		particle_positions, particle_sizes = entity.rigidbody.GetParticleData()

		particle_indices = (particle_positions * self.indices_per_meter)
		particle_indices = particle_indices.int()

		render_space = self.space.clone()
		render_space[particle_indices[:, 0], particle_indices[:, 1], particle_indices[:, 2]] = entity.rigidbody.material_vals
		
		direction_hint_indices = (entity.rigidbody.body_origin[:, [0, 1]] + entity.desired_direction) * 10
		direction_hint_indices = direction_hint_indices.int()
		z_index = (entity.rigidbody.body_origin[0, 2] * 10).int()
		render_space[direction_hint_indices[0, 0], direction_hint_indices[0, 1], z_index] = 5
		
		return render_space
		