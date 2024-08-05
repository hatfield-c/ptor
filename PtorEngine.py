import cv2

import Rigidbody
import Renderer

class PtorEngine:
	def __init__(self):
		self.renderer = Renderer.Renderer()
		self.rigidbody = Rigidbody.Rigidbody()
	
	def InstantiateScenario(self):
		pass
	
	def Run(self):
		
		for i in range(1):
			
			
			render_frame = self.renderer.Render(self.rigidbody)
			render_frame = render_frame.numpy()
			
			cv2.imshow("PtorEngine v0.0.1 Render", render_frame)
			cv2.waitKey(0)