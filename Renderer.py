import math
import matplotlib.pyplot as plt
import torch

import Quaternion

class Renderer:
	def __init__(self):
		self.render_width = 256#5
		self.render_height = 256#3
		self.resolution = (self.render_height, self.render_width)
		self.render_fov = torch.FloatTensor([math.pi / 2, math.pi / 2])
		
		self.min_render_distance = 0.05
		self.max_render_distance = 100
		
		self.render_location = torch.FloatTensor([[2, -1, 2]])
		
		lateral_displacement = torch.sin(self.render_fov[1] / 2)
		vertical_displacement = torch.sin(self.render_fov[0] / 2)
		
		x_values = torch.linspace(-lateral_displacement, lateral_displacement, self.resolution[1])
		z_values = torch.linspace(-vertical_displacement, vertical_displacement, self.resolution[0])
		
		xz_origins = torch.meshgrid([x_values, z_values], indexing = "xy")
		xz_origins = torch.stack(xz_origins)
		xz_origins = xz_origins.reshape(2, -1).T
		
		ray_origins = (xz_origins, torch.ones((xz_origins.shape[0], 1)))
		ray_origins = torch.cat(ray_origins, dim = 1)
		ray_origins = ray_origins[:, [0, 2, 1]]
		ray_origins= ray_origins / torch.norm(ray_origins, dim = 1, keepdim = True)
		
		self.ray_origins = ray_origins * self.min_render_distance
		self.ray_endgins = ray_origins * self.max_render_distance
		
	def Render(self, rigidbody):
		canvas = torch.zeros(self.resolution)
		
		ray_origins, ray_endgins = self.UpdateRayPoints()
		
		particles, particle_sizes = rigidbody.GetParticleData()
		
		ray_vectors = ray_endgins - ray_origins
		ray_vectors = ray_vectors / torch.norm(ray_vectors, dim = 1, keepdim = True)
		
		ray_origins = ray_origins.unsqueeze(2)
		ray_vectors = ray_vectors.unsqueeze(2)
		particles = particles.unsqueeze(2)
		particle_sizes = particle_sizes.unsqueeze(0)
		
		particles = particles.repeat((1, 1, ray_origins.shape[0]))
		particles = particles.transpose(0, 2)
		
		particle_vectors = particles - ray_origins
		particle_distances = particle_vectors * ray_vectors
		particle_distances = torch.sum(particle_distances, dim = 1, keepdim = True)
		
		particle_projections = particle_distances * ray_vectors
		particle_projections = particle_projections + ray_origins
		
		ray_offsets = particle_projections - particles
		ray_offsets = torch.norm(ray_offsets, dim = 1)
		
		shading_val = particle_sizes - ray_offsets
		shading_val = torch.clamp_min(shading_val, 0) / particle_sizes
		
		hitmask = torch.ceil(shading_val)
		invert_hitmask = (1 - hitmask) * 1e10
		shading_val = shading_val ** 0.5
		
		max_values, max_indices = torch.max(shading_val, dim = 1)
		
		particle_distances = particle_distances[:, 0]
		closest_hits = invert_hitmask + particle_distances
		
		closest_values, closest_indices = torch.min(closest_hits, dim = 1)
		
		canvas = shading_val[torch.arange(shading_val.shape[0]), closest_indices]
		canvas = canvas.reshape(self.resolution)
		#canvas = max_values.reshape(self.resolution)
		canvas = canvas.flip(dims = (0,))
		
		print(ray_origins.shape)
		print(ray_vectors.shape)
		print("")
		print(particles.shape)
		print(particle_vectors.shape)
		print("")
		print(particle_distances.shape)
		print(particle_projections.shape)
		print("")
		print(ray_offsets.shape)
		print(shading_val.shape)
		print(invert_hitmask.shape)
		print(max_values.shape)
		
		#exit()
		
		return canvas
	
	def UpdateRayPoints(self):
		pitch_rotation = torch.FloatTensor([[0.924, -0.383, 0, 0]])
		yaw_rotation = torch.FloatTensor([[-0.924, 0, 0, -0.383]])
		
		ray_origins = Quaternion.RotatePoints(self.ray_origins, pitch_rotation)
		ray_origins = Quaternion.RotatePoints(ray_origins, yaw_rotation)
		ray_origins += self.render_location
		
		ray_endgins = Quaternion.RotatePoints(self.ray_endgins, pitch_rotation)
		ray_endgins = Quaternion.RotatePoints(ray_endgins, yaw_rotation)
		ray_endgins += self.render_location
		
		return ray_origins, ray_endgins
		
