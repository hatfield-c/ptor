import cv2
import numpy as np
import torch
import time

import CONFIG
import Rigidbody
import WorldSpace
import RenderCamera
import Quaternion

class PtorEngine:
	def __init__(self):
		self.camera = RenderCamera.RenderCamera()
		self.rigidbody = Rigidbody.Rigidbody()
		self.world_space = WorldSpace.WorldSpace()
		
		self.avg_fps = 1
		self.delta_time = CONFIG.delta_time
		
		self.gravity = torch.FloatTensor([[0, 0, -9.8]]).cuda()
		self.gravity_delta = self.gravity * self.delta_time
		
		self.time_step = 0
	
	def InstantiateScenario(self):
		pass
	
	def Run(self):
		
		for i in range(10000000):
			start_time = time.time()	
			
			self.PhysicsUpdate()
			self.ScenarioUpdate()
			self.RenderUpdate(start_time)
			
			self.time_step = i
			
	def PhysicsUpdate(self):
		self.rigidbody.Accelerate(self.gravity_delta)

		for i in range(4):
			thrust = torch.FloatTensor([[0, 0, 0.28]]).cuda()
			motor_position = self.rigidbody.motor_positions[[i]]
			
			thrust = Quaternion.RotatePoints(thrust, self.rigidbody.body_rotation)
			motor_position = Quaternion.RotatePoints(motor_position, self.rigidbody.body_rotation)
			
			self.rigidbody.AddForce(thrust, motor_position)
		
		self.rigidbody.Update()
			
	def ScenarioUpdate(self):
		self.camera.RotateAroundAnchor(torch.FloatTensor([460, 460, 10]).cuda(), -0.01 * 0)
		
	def RenderUpdate(self, start_time):
		self.camera.Follow(self.rigidbody)
		
		render_frame = self.camera.CaptureImage(self.world_space, self.rigidbody)
		render_frame = render_frame.cpu().numpy().astype(np.uint8)
		
		render_frame = cv2.resize(render_frame, (1024, 1024), interpolation = cv2.INTER_NEAREST)
		self.DrawFps(render_frame, self.avg_fps)
		
		cv2.imshow("PtorEngine v0.0.1 Render", render_frame)

		elapsed_time = time.time() - start_time
		
		time_wait = 0
		
		if elapsed_time < self.delta_time:
			time_diff = self.delta_time - elapsed_time
			time_wait = int(time_diff * 1000)
			
		if time_wait < 1:
			time_wait = 1
		
		cv2.waitKey(time_wait)
			
		elapsed_time = time.time() - start_time
		fps = 1 / elapsed_time
		self.avg_fps = ((9 / 10) * self.avg_fps) + (fps / 10)
			
	def DrawFps(self, canvas, fps):
		fps = "{:.2f}".format(fps)
		cv2.putText(canvas, fps, (20, 40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 0, 0), thickness = 3)
		cv2.putText(canvas, fps, (20, 40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 255, 255), thickness = 2)