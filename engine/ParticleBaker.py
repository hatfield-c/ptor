import torch

import CONFIG

class ParticleBaker:
	def __init__(self, particle_path):
		particle_data = torch.load(CONFIG.rigid_drone_particles_path)
		
		self.particles = particle_data[:, 0:3]
		self.materials = particle_data[:, [3]]
		self.sizes = particle_data[:, [4]]
		self.masses = particle_data[:, [5]]
		
	def BakeRigidbodyData(self, save_path):
		body_data = {}
		
		body_data["center_of_mass"] = self.GetMassCenter()
		body_data["inertia_moment"] = self.GetInertiaMoment(body_data["center_of_mass"])
		body_data["total_mass"] = torch.sum(self.masses)
		
		torch.save(body_data, save_path)
		
	def BakeEnvironmentData(self, save_path):
		baked_data = {}
		
		torch.save(baked_data, save_path)
		
	def GetInertiaMoment(self, center_of_mass):
		positions = self.particles - center_of_mass
		
		inertia_moment = torch.zeros((3, 3))
		for i in range(positions.shape[0]):
			print(i)
			particle_inertia = torch.FloatTensor([
				[
					 self.masses[i] * ((positions[i, 1] ** 2) + (positions[i, 2] ** 2)),
					 self.masses[i] * (-positions[i, 0] * positions[i, 1]),
					 self.masses[i] * (-positions[i, 0] * positions[i, 2])
				],
				[
					 self.masses[i] * (-positions[i, 1] * positions[i, 0]),
					 self.masses[i] * ((positions[i, 0] ** 2) + (positions[i, 2] ** 2)),
					 self.masses[i] * (-positions[i, 1] * positions[i, 2])
				],
				[
					 self.masses[i] * (-positions[i, 2] * positions[i, 0]),
					 self.masses[i] * (-positions[i, 2] * positions[i, 1]),
					 self.masses[i] * ((positions[i, 0] ** 2) + (positions[i, 1] ** 2))
				]
			])
			
			inertia_moment += particle_inertia
			
		return inertia_moment
		
	def GetMassCenter(self):
		center_of_mass = self.particles * self.masses
		center_of_mass = torch.sum(center_of_mass, dim = 0, keepdim = True) / torch.sum(self.masses)
		
		return center_of_mass