import torch
import math
import numpy as np
import trimesh

from collections import OrderedDict

class MeshConverter:
	def __init__(self, mesh_path):

		scene = trimesh.load(mesh_path, force = "scene", group_material = False)
		mesh_list = scene.geometry
		
		if not isinstance(mesh_list, OrderedDict):
			meshes_dict = OrderedDict()
			meshes_dict["mesh"] = mesh_list
			mesh_list = meshes_dict
		
		self.mesh_list = mesh_list
	
	def SaveAsVoxelParticles(self, save_path, voxel_size, expected_mass = None):
		
		particle_data_list = []

		iterations = 0
		for mesh_name in self.mesh_list:
			completion = int(100 * iterations / len(self.mesh_list))
			iterations += 1
			print(str(completion) + "% -", mesh_name)
			
			mesh = self.mesh_list[mesh_name]
			matrix_list = self.GetMatrixList(mesh, voxel_size)
			
			is_contained = mesh.contains(matrix_list)
			particle_points = matrix_list[is_contained].cuda()
			
			if particle_points.shape[0] < 1:
				print("    <Warning>:", mesh_name, "has particle count of 0. Ignoring mesh.")
				continue
			
			material_name = mesh.visual.material.name
			material_index = int(material_name[4:])
			material_index = torch.FloatTensor([[material_index]]).cuda()
			material_index = material_index.repeat((particle_points.shape[0], 1))
			
			size = voxel_size + torch.zeros((particle_points.shape[0], 1)).cuda()
			mass = torch.zeros((particle_points.shape[0], 1)).cuda()
			
			if "size" in mesh.metadata:
				size = torch.FloatTensor([mesh.metadata["size"]]).view(1, 1).cuda()
				size = size.repeat((particle_points.shape[0], 1))
				
			if "mass" in mesh.metadata:
				mass = torch.FloatTensor([mesh.metadata["mass"] / particle_points.shape[0]]).view(1, -1).cuda()
				mass = mass.repeat((particle_points.shape[0], 1))
				
			particle_data = torch.cat([particle_points, material_index, size, mass], dim = 1)
			particle_data_list.append(particle_data)

		save_data = torch.cat(particle_data_list, dim = 0)
		total_particles = save_data.shape[0]
		total_mass = torch.sum(save_data[:, 5])
		
		print("[Total Particles] :", total_particles)
		print("[Total Mass]      :", total_mass.cpu().item())
		print("[Expected Mass]   :", expected_mass)
		
		if expected_mass is not None and expected_mass != total_mass:
			mass_diff = torch.abs(expected_mass - total_mass)
			mass_assignment = mass_diff / total_particles
			diff_ratio = int(100 * (mass_diff / expected_mass))
			
			save_data[:, [5]] = save_data[:, [5]] + mass_assignment
			repaired_mass = torch.sum(save_data[:, 5])
			
			print("[Repaired Mass]:", repaired_mass.cpu().item())
			print("    [Error Mass] :", mass_diff.cpu().item())
			print("    [Error Ratio]:", str(diff_ratio) + "%")
		
		torch.save(save_data, save_path)
	
	def GetMatrixList(self, mesh, voxel_size):
		lower_bounds = mesh.bounds[0]
		upper_bounds = mesh.bounds[1]
		
		x_steps = int((upper_bounds[0] - lower_bounds[0]) / voxel_size)
		y_steps = int((upper_bounds[1] - lower_bounds[1]) / voxel_size)
		z_steps = int((upper_bounds[2] - lower_bounds[2]) / voxel_size)
		
		x = torch.linspace(lower_bounds[0], upper_bounds[0], x_steps + 1)
		y = torch.linspace(lower_bounds[1], upper_bounds[1], y_steps + 1)
		z = torch.linspace(lower_bounds[2], upper_bounds[2], z_steps + 1)
		
		matrix_list = torch.meshgrid([x, y, z], indexing = "xy")
		matrix_list = torch.stack(matrix_list)
		matrix_list = matrix_list.reshape(3, -1).T
		
		print("    [Matrix Size] :", matrix_list.shape[0])
		print("    [Lower Bounds]:", lower_bounds)
		print("    [Upper Bounds]:", upper_bounds)
		
		return matrix_list
	
	def SaveAsObjectParticles(self, save_path):
		particle_data_list = []
		
		for mesh_name in self.mesh_list:
			mesh = self.mesh_list[mesh_name]
			
			material_name = mesh.visual.material.name
			material_index = int(material_name[4:])
			
			position = torch.FloatTensor(mesh.center_mass).view(1, -1)
			material_index = torch.FloatTensor([material_index]).view(1, -1)
			
			size = torch.ones((1, 1))
			mass = torch.zeros((1, 1))
			
			if "size" in mesh.metadata:
				size = torch.FloatTensor([mesh.metadata["size"]]).view(1, 1)
				
			if "mass" in mesh.metadata:
				mass = torch.FloatTensor([mesh.metadata["mass"]]).view(1, 1)
			
			particle_data = torch.cat((position, material_index, size, mass), dim = 1)
			particle_data_list.append(particle_data)
			
		save_data = torch.cat(particle_data_list, dim = 0)
		
		torch.save(save_data, save_path)
	
	def ConvertToParticles(self):
		pass
	