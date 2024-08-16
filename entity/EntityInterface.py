
class EntityInterface:
	def __init__(self):
		pass
	
	def Update(self):
		raise NotImplementedError(self.__class__.__name__ + ' is not implemented.')