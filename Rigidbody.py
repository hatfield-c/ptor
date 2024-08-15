import torch

import CONFIG
import Quaternion

class Rigidbody:
	def __init__(self):
		
		physics_vals = self.LoadPhysicsValues()
		
		particle_data = torch.load(CONFIG.visual_drone_particles_path)
		particle_positions = particle_data[:, :3]
		
		self.material_vals = particle_data[:, 3].cuda()
		
		self.particle_positions = particle_positions.cuda()
		self.alpha_positions = particle_positions.clone().cuda()
		
		self.particle_count = self.particle_positions.shape[0]
		self.particle_dimensionality = self.particle_positions.shape[1]
		
		self.particle_sizes = torch.ones(self.particle_count).cuda()
		self.particle_masses = torch.ones(self.particle_count).cuda()
		
		self.body_velocity = 0
		self.body_mass = torch.sum(self.particle_masses)
		
		self.center_of_mass = physics_vals["center_of_mass"]
		self.inertia_moment = physics_vals["inertia_moment"]
		
		self.alpha_positions = self.alpha_positions - self.center_of_mass
		
		world_origin = torch.FloatTensor([[46, 46, 2]]).cuda() 
		self.body_origin = self.center_of_mass + world_origin
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
		
	def LoadPhysicsValues(self):
		print("Loading physics values...")
		particle_data = torch.load(CONFIG.rigid_drone_particles_path)
		
		positions = particle_data[:, 0:3]
		mass = particle_data[:, [5]]
		
		center_of_mass = positions * mass
		center_of_mass = torch.sum(center_of_mass, dim = 0, keepdim = True) / torch.sum(mass)
		
		positions = positions - center_of_mass
		
		inertia_moment = torch.zeros((3, 3))
		for i in range(positions.shape[0]):
			particle_inertia = torch.FloatTensor([
				[
					 mass[i] * ((positions[i, 1] ** 2) + (positions[i, 2] ** 2)),
					 mass[i] * (-positions[i, 0] * positions[i, 1]),
					 mass[i] * (-positions[i, 0] * positions[i, 2])
				],
				[
					 mass[i] * (-positions[i, 1] * positions[i, 0]),
					 mass[i] * ((positions[i, 0] ** 2) + (positions[i, 2] ** 2)),
					 mass[i] * (-positions[i, 1] * positions[i, 2])
				],
				[
					 mass[i] * (-positions[i, 2] * positions[i, 0]),
					 mass[i] * (-positions[i, 2] * positions[i, 1]),
					 mass[i] * ((positions[i, 0] ** 2) + (positions[i, 1] ** 2))
				]
			])
			
			inertia_moment += particle_inertia
			
		physics_values = {
			"center_of_mass": center_of_mass,
			"inertia_moment": inertia_moment
		}
			
		print("    Loaded!")
		
		return physics_values
		
	def GetParticleData(self):

		return self.particle_positions, self.particle_sizes