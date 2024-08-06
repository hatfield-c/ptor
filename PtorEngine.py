import cv2
import torch
import time

import Rigidbody
import RenderCamera
import Quaternion

class PtorEngine:
	def __init__(self):
		self.camera = RenderCamera.RenderCamera()
		self.rigidbody = Rigidbody.Rigidbody()
		
		self.avg_fps = 1
		self.delta_time = 1 / 30
	
	def InstantiateScenario(self):
		pass
	
	def Run(self):
		
		for i in range(10000000):
			start_time = time.time()	
			
			self.PhysicsUpdate()
			self.ScenarioUpdate()
			self.RenderUpdate(start_time)
			
			
	def PhysicsUpdate(self):
		pass
		self.rigidbody.Update()
			
	def ScenarioUpdate(self):
		self.camera.RotateAroundAnchor(torch.FloatTensor([0, 0, 0]).cuda(), 0.01)
		
	def RenderUpdate(self, start_time):
		render_frame = self.camera.CaptureImage(self.rigidbody)
		render_frame = render_frame.cpu().numpy()
		
		self.DrawFps(render_frame, self.avg_fps)
		
		cv2.imshow("PtorEngine v0.0.1 Render", render_frame)

		elapsed_time = time.time() - start_time
		
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
		cv2.putText(canvas, fps, (20, 20), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = 0, thickness = 2)
		cv2.putText(canvas, fps, (20, 20), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = 1, thickness = 1)