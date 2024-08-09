import torch

import CONFIG
import Quaternion

class Rigidbody:
	def __init__(self):
		x = torch.linspace(0, 1, 20)
		y = torch.linspace(0, 1, 20)
		z = torch.linspace(0, 1, 20)
		
		position_mesh = torch.meshgrid([x, y, z], indexing = "xy")
		position_mesh = torch.stack(position_mesh)
		position_mesh = position_mesh.reshape(3, -1).T
		
		self.particle_positions = position_mesh.cuda()
		self.alpha_positions = position_mesh.clone().cuda()
		
		self.particle_count = self.particle_positions.shape[0]
		self.particle_dimensionality = self.particle_positions.shape[1]
		
		self.particle_sizes = torch.ones(self.particle_count).cuda()
		self.particle_masses = torch.ones(self.particle_count).cuda()
		
		self.body_velocity = 0
		self.body_mass = torch.sum(self.particle_masses)
		
		normalized_masses = self.particle_masses / torch.sum(self.particle_masses)
		weighted_positions = self.particle_positions * normalized_masses.reshape((self.particle_count, 1))
		center_of_mass = torch.sum(weighted_positions, dim = 0, keepdim = True).cuda()
		
		self.alpha_positions = self.alpha_positions - center_of_mass
		
		world_origin = torch.FloatTensor([[48, 41, 2]]).cuda() 
		self.body_origin = center_of_mass + world_origin
		self.body_rotation = torch.FloatTensor([[0, 0, 0, 1]]).cuda()
		self.body_velocity = torch.FloatTensor([[0, 0, 0]]).cuda()
		self.body_angular_velocity = torch.FloatTensor([[1, 1, 0.33]]).cuda()
		
	def Update(self):
		position_delta = self.body_velocity * CONFIG.delta_time
		self.body_origin = self.body_origin + position_delta
		
		rotation_angles_delta = self.body_angular_velocity * CONFIG.delta_time
		quaternion_delta = Quaternion.QuaternionFromEulerAngles(rotation_angles_delta[0]).cuda()
		self.body_rotation = Quaternion.MultiplyQuaternions(quaternion_delta, self.body_rotation)
		
		self.particle_positions = Quaternion.RotatePoints(self.alpha_positions, self.body_rotation)
		self.particle_positions = self.particle_positions + self.body_origin
		
	def GetParticleData(self):

		return self.particle_positions, self.particle_sizes