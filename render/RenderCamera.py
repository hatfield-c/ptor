import math
import matplotlib.pyplot as plt
import torch
import torch
import cv2
import time

import CONFIG
import engine.Quaternion as Quaternion

class RenderCamera:
	def __init__(self):
		self.render_width = 240
		self.render_height = 240
		self.resolution = (self.render_height, self.render_width)
		self.render_fov = torch.FloatTensor([math.pi / 2, math.pi / 2])
		
		self.pitch_rotation = Quaternion.QuaternionFromEulerParams([1, 0, 0], -math.pi / 6).cuda()
		self.yaw_rotation = Quaternion.QuaternionFromEulerParams([0, 0, 1], 0).cuda()
		
		self.min_render_distance = 0.05
		self.max_render_distance = 350
		
		self.camera_position = torch.FloatTensor([[460, 450, 27]]).cuda()
		self.camera_offset = torch.FloatTensor([[0, -15, 7]]).cuda()
		
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
		
		self.ray_origins = ray_origins.cuda() * self.min_render_distance
		self.ray_endgins = ray_origins.cuda() * self.max_render_distance
		
		self.ray_steps = 750
		self.ray_delta = torch.linspace(0, 1, self.ray_steps).reshape(1, -1, 1).cuda()
		
		self.static_query_indices = torch.arange(self.ray_origins.shape[0]).cuda()
		self.mat_library = torch.FloatTensor([
			[0, 0, 0],
			[180, 180, 180],
			[255, 0, 0],
			[0, 255, 0],
			[0, 0, 255],
			[255, 255, 0],
			[255, 0, 255],
			[0, 255, 255],
			[100, 100, 100],
			[140, 140, 140],
			[0, 45, 0],
		]).cuda()
	
	def Follow(self, rigidbody):
		target_position = rigidbody.body_origin.reshape(1, -1) * CONFIG.indices_per_meter
		target_rotation = rigidbody.body_rotation.clone()
		
		yaw_rotation = target_rotation.clone()
		yaw_rotation[0, [1]] = 0
		yaw_rotation[0, [2]] = 0
		yaw_norm = torch.linalg.norm(yaw_rotation)
		
		if yaw_norm > 0:
			yaw_rotation = yaw_rotation / yaw_norm
		
		camera_offset = Quaternion.RotatePoints(self.camera_offset, yaw_rotation)
		
		self.camera_position = target_position + camera_offset
		
		target_rotation[0, [1]] = 0
		target_rotation[0, [2]] = 0
		target_rotation = target_rotation / torch.linalg.norm(target_rotation)
		self.yaw_rotation = target_rotation
	
	def CaptureImage(self, world_space, rigidbody):
		render_space = world_space.GetRenderSpace(rigidbody)
		ray_origins, ray_endgins = self.UpdateRayPoints()
		
		position_queries = ((1 - self.ray_delta) * ray_origins.unsqueeze(1)) + (self.ray_delta* ray_endgins.unsqueeze(1))
		position_queries = position_queries.view(-1, 3)
		position_queries = torch.clamp(position_queries, torch.zeros((1, 3)).cuda(), torch.FloatTensor(list(world_space.space.shape)).view(1, -1).cuda() - 1)
		index_queries = position_queries.int()
		
		query_vals = render_space[index_queries[:, 0], index_queries[:, 1], index_queries[:, 2]]
		query_vals = query_vals.view(ray_origins.shape[0], self.ray_steps)
		
		is_occupied = (query_vals > 0)
		_, nearest_occupied_indices = torch.max((is_occupied.cumsum(1) == 1) & is_occupied, dim = 1)
		
		render_ids = query_vals[self.static_query_indices, nearest_occupied_indices].int()
		canvas = self.mat_library[render_ids]
		
		shading = torch.exp(-0.004 * nearest_occupied_indices)
		
		shading = shading.view(-1, 1)
		
		canvas = canvas * shading
		
		canvas = canvas.view((self.render_height, self.render_width, 3))
		canvas = canvas.flip(dims = (0,))
		
		return canvas
	
	def UpdateRayPoints(self):

		ray_origins = Quaternion.RotatePoints(self.ray_origins, self.pitch_rotation)
		ray_origins = Quaternion.RotatePoints(ray_origins, self.yaw_rotation)
		ray_origins += self.camera_position
		
		ray_endgins = Quaternion.RotatePoints(self.ray_endgins, self.pitch_rotation)
		ray_endgins = Quaternion.RotatePoints(ray_endgins, self.yaw_rotation)
		ray_endgins += self.camera_position

		return ray_origins, ray_endgins
	
	def RotateAroundAnchor(self, anchor, theta):
		camera_position = self.camera_position - anchor
		
		quaternion = Quaternion.QuaternionFromEulerParams([0, 0, 1], theta).cuda()
		self.camera_position = Quaternion.RotatePoints(camera_position, quaternion) + anchor
		
		self.yaw_rotation = Quaternion.MultiplyQuaternions(self.yaw_rotation, quaternion)
