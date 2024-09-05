import torch
import math
import time

import engine.Quaternion as Quaternion
import engine.Transform as Transform
import entity.sensor.SensorInterface as SensorInterface

class DepthSensor(SensorInterface.SensorInterface):
	def __init__(
			self,
			track_time = False
		):
		self.track_time = track_time
		
		self.avg_render_time = 0
		
		self.render_width = 424
		self.render_height = 240
		self.resolution = (self.render_height, self.render_width)
		self.render_fov = torch.FloatTensor([0.977, 1.5]).cuda()
		
		self.min_render_distance = 0.005
		self.max_render_distance = 20
		
		self.ray_steps = 750
		self.ray_delta = torch.linspace(0, 1, self.ray_steps).reshape(1, -1, 1).cuda()
		
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
		self.static_query_indices = torch.arange(self.ray_origins.shape[0]).cuda()
		
	def ReadSensorImage(self, world_space, position, quaternion):
		depth_vals = self.ReadSensorFiltered(world_space, position, quaternion)
		depth_vals = depth_vals / self.max_render_distance
		depth_vals = depth_vals.flip(dims = (0,)) * 255
		
		return depth_vals
		
	def ReadSensorFiltered(self, world_space, position, quaternion):
		depth_vals = self.ReadSensor(world_space, position, quaternion)
		
		start_column = int((self.render_width - self.render_height) / 2)
		end_column = self.render_width - start_column
		
		depth_vals = depth_vals.view((self.render_height, self.render_width, 1))
		depth_vals = depth_vals[:, start_column:end_column]
		
		return depth_vals
		
	def ReadSensor(self, world_space, position, quaternion):
		
		if self.track_time:
			start_time = time.time()		
		
		ray_origins, ray_endgins = self.UpdateRayPoints(position, quaternion, world_space.indices_per_meter)
		
		position_queries = ((1 - self.ray_delta) * ray_origins.unsqueeze(1)) + (self.ray_delta* ray_endgins.unsqueeze(1))
		position_queries = position_queries.view(-1, 3)
		position_queries = torch.clamp(position_queries, torch.zeros((1, 3)).cuda(), torch.FloatTensor(list(world_space.space.shape)).view(1, -1).cuda() - 1)
		index_queries = position_queries.int()
		position_queries = position_queries.view(-1, self.ray_steps, 3)
		
		query_vals = world_space.space[index_queries[:, 0], index_queries[:, 1], index_queries[:, 2]]
		query_vals = query_vals.view(ray_origins.shape[0], self.ray_steps)
		
		is_occupied = (query_vals > 0)
		_, nearest_occupied_indices = torch.max((is_occupied.cumsum(1) == 1) & is_occupied, dim = 1)
		
		depth_vals = position_queries[self.static_query_indices, nearest_occupied_indices] - (position * world_space.indices_per_meter)
		depth_vals = torch.linalg.norm(depth_vals, dim = 1) / world_space.indices_per_meter
		
		if self.track_time:
			time_diff = time.time() - start_time
			self.avg_render_time = (self.avg_render_time + time_diff) / 2

		return depth_vals
	
	def UpdateRayPoints(self, position, quaternion, world_scaling = None):
		ray_origins = self.ray_origins
		ray_endgins = self.ray_endgins
		
		if world_scaling is not None:
			position = position * world_scaling
			ray_origins = ray_origins * world_scaling
			ray_endgins = ray_endgins * world_scaling
		
		ray_origins = Quaternion.RotatePoints(ray_origins, quaternion)
		ray_origins += position
		
		ray_endgins = Quaternion.RotatePoints(ray_endgins, quaternion)
		ray_endgins += position

		return ray_origins, ray_endgins