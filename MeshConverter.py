import torch
import trimesh
from collections import OrderedDict

class MeshConverter:
	def __init__(self, mesh_path):

		file_obj, file_type, metadata, opened, resolver = trimesh.exchange.load._parse_file_args(mesh_path, "obj")

		scene = trimesh.load(mesh_path, force = "scene", group_material = False)
		
		mesh_list = scene.geometry
		
		if not isinstance(mesh_list, OrderedDict):
			meshes_dict = OrderedDict()
			meshes_dict["mesh"] = mesh_list
			mesh_list = meshes_dict
			
		self.mesh_list = mesh_list
	
	def SaveAsVoxelParticles(self, save_path, particle_size, particle_mass):
		
		particle_data_list = []
		
		for mesh_name in self.mesh_list:
			mesh = self.mesh_list[mesh_name]
			voxelized = mesh.voxelized(particle_size, max_iter = 10000000000)
			voxelized = voxelized.hollow()
			
			material_name = mesh.visual.material.name
			material_index = int(material_name[4:])
			
			particle_points = torch.FloatTensor(voxelized.points).cuda()
			material_vals = torch.FloatTensor([[material_index]]).cuda()
			material_vals = material_vals.repeat((particle_points.shape[0], 1))
			
			particle_data = torch.cat([particle_points, material_vals], dim = 1)
			particle_data_list.append(particle_data)

		save_data = torch.cat(particle_data_list, dim = 0)
		
		sizes = torch.FloatTensor([[particle_size]]).cuda()
		masses = torch.FloatTensor([[particle_mass]]).cuda()
		
		sizes = sizes.repeat((save_data.shape[0], 1))
		masses = masses.repeat((save_data.shape[0], 1))
		
		save_data = torch.cat((save_data, sizes, masses), dim = 1)
		
		torch.save(save_data, save_path)
	
	def SaveAsObjectParticles(self, save_path):
		particle_data_list = []
		
		for material in self.mesh_list:
			mesh = self.mesh_list[material]
			
			print(material, mesh)
	
	def ConvertToParticles(self):
		pass
	