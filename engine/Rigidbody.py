import torch
import math

import CONFIG
import engine.Transform as Transform
import engine.Quaternion as Quaternion

class Rigidbody:
	def __init__(self):
		
		physics_vals = torch.load(CONFIG.rigid_drone_physics_path)
		
		particle_data = torch.load(CONFIG.visual_drone_particles_path)
		particle_positions = particle_data[:, :3]
		
		self.material_vals = particle_data[:, 3].cuda()
		
		self.particle_positions = particle_positions.cuda()
		self.alpha_positions = particle_positions.clone().cuda()
		
		self.particle_count = self.particle_positions.shape[0]
		self.particle_dimensionality = self.particle_positions.shape[1]
		
		self.particle_sizes = torch.ones(self.particle_count).cuda()
		self.particle_masses = torch.ones(self.particle_count).cuda()
		
		self.center_of_mass = physics_vals["center_of_mass"]
		self.inertia_moment = physics_vals["inertia_moment"].cuda()
		self.body_mass = physics_vals["total_mass"]
		
		self.alpha_positions = self.alpha_positions - self.center_of_mass
		
		world_origin = torch.FloatTensor([[45, 10, 2]]).cuda() 
		#world_origin = torch.FloatTensor([[62, 53, 2]]).cuda() 
		self.body_origin = self.center_of_mass + world_origin
		self.body_rotation = Quaternion.QuaternionFromEulerAngles([0, 0, 0]).cuda()
		#self.body_rotation = Quaternion.QuaternionFromEulerAngles([-math.pi / 12, 0, 0]).cuda()
		self.body_velocity = torch.FloatTensor([[0, 0, 0]]).cuda()
		self.body_angular_velocity = torch.FloatTensor([[0, 0, 0]]).cuda() * 1
		self.inverse_inertia = self.GetInverseIntertia()
		
	def Update(self):
		position_delta = self.body_velocity * CONFIG.delta_time
		self.body_origin = self.body_origin + position_delta
		
		rotation_angles_delta = self.body_angular_velocity * CONFIG.delta_time
		quaternion_delta = Quaternion.QuaternionFromEulerAngles(rotation_angles_delta[0]).cuda()
		self.body_rotation = Quaternion.MultiplyQuaternions(quaternion_delta, self.body_rotation)
		
		self.particle_positions = Quaternion.RotatePoints(self.alpha_positions, self.body_rotation)
		self.particle_positions = self.particle_positions + self.body_origin
		
		self.inverse_inertia = self.GetInverseIntertia()
		
	def Accelerate(self, acceleration):
		#return
		self.body_velocity += acceleration * CONFIG.delta_time
		
	def AddForce(self, force, displacement):
		#return
		torque = torch.linalg.cross(displacement, force).view(-1, 1)
		self.AddTorque(torque)
		
		acceleration = force / self.body_mass	
		acceleration = acceleration.view(1, -1)
		
		self.body_velocity += (acceleration * CONFIG.delta_time)
	
	def AddTorque(self, torque):
		#return
		angular_velocity_delta = torch.matmul(self.inverse_inertia, torque)
		
		self.body_angular_velocity += (angular_velocity_delta.T * CONFIG.delta_time)
	
	def AirResistance(self, wind_vector):
		#return
		viscous_torque = 6.17e-5 * self.body_angular_velocity
		self.AddTorque(-viscous_torque.T)
		
		velocity_direction = self.body_velocity
		velocity_speed = torch.linalg.norm(velocity_direction, dim = 1)
		if velocity_speed > 0:
			velocity_direction = velocity_direction / velocity_speed
		
		wind_direction = wind_vector
		wind_speed = torch.linalg.norm(wind_direction, dim = 1)
		if wind_speed > 0:
			wind_direction = wind_direction / wind_speed
		
		velocity_drag_force = 0.5 * (1.293e-3) * 0.47 * math.pi * (0.25 ** 2) * velocity_speed * -velocity_direction
		wind_drag_force = 0.5 * (1.293e-3) * 0.47 * math.pi * (0.25 ** 2) * wind_speed * wind_direction
		
		self.AddForce(velocity_drag_force, Transform.ZEROS)
		self.AddForce(wind_drag_force, Transform.ZEROS)
	
	def GetInverseIntertia(self):
		rotation_matrix = Quaternion.MatrixFromQuaternion(self.body_rotation)
		
		inverse_inertia = torch.linalg.solve(self.inertia_moment, rotation_matrix.T)
		inverse_inertia = torch.matmul(rotation_matrix, inverse_inertia)
		
		return inverse_inertia
		
	def GetParticleData(self):

		return self.particle_positions, self.particle_sizes