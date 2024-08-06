import torch
import math

def RotatePoints(points, quaternion):
	point_quat = torch.zeros((points.shape[0], 1)).cuda()
	point_quat = torch.cat((point_quat, points), dim = 1)
	
	quaternion_conjugate = GetQuaternionConjugate(quaternion)
	
	rotated_points = MultiplyQuaternions(quaternion, point_quat, is_normalized = False)
	rotated_points = MultiplyQuaternions(rotated_points, quaternion_conjugate, is_normalized = False)

	return rotated_points[:, 1:]

def MultiplyQuaternions(q0, q1, is_normalized = True):

	result = torch.stack((
		(q0[:, 0] * q1[:, 0]) - (q0[:, 1] * q1[:, 1]) - (q0[:, 2] * q1[:, 2]) - (q0[:, 3] * q1[:, 3]),
		(q0[:, 0] * q1[:, 1]) + (q0[:, 1] * q1[:, 0]) + (q0[:, 2] * q1[:, 3]) - (q0[:, 3] * q1[:, 2]),
		(q0[:, 0] * q1[:, 2]) - (q0[:, 1] * q1[:, 3]) + (q0[:, 2] * q1[:, 0]) + (q0[:, 3] * q1[:, 1]),
		(q0[:, 0] * q1[:, 3]) + (q0[:, 1] * q1[:, 2]) - (q0[:, 2] * q1[:, 1]) + (q0[:, 3] * q1[:, 0])
	), dim = 1)
			
	if is_normalized:
		result = result / torch.norm(result, dim = 1, keepdim = True)
	
	return result

def GetQuaternionConjugate(quaternion):
	conjugate = torch.stack([quaternion[:, 0], -quaternion[:, 1], -quaternion[:, 2], -quaternion[:, 3]], dim = 1)
	
	return conjugate

def QuaternionFromEulerParams(axis, angle):
	sin_val = math.sin(angle / 2)
	
	quaternion = torch.FloatTensor([[
		math.cos(angle / 2), 
		sin_val * axis[0],
		sin_val * axis[1],
		sin_val * axis[2],
	]])
	
	return quaternion

def QuaternionFromEulerAngles(angles):
	alpha = angles[0] / 2
	beta = angles[1] / 2
	gamma = angles[2] / 2
	
	quaternion = torch.FloatTensor([[
		(math.cos(alpha) * math.cos(beta) * math.cos(gamma)) + (math.sin(alpha) * math.sin(beta) * math.sin(gamma)),
		(math.sin(alpha) * math.cos(beta) * math.cos(gamma)) - (math.cos(alpha) * math.sin(beta) * math.sin(gamma)),
		(math.cos(alpha) * math.sin(beta) * math.cos(gamma)) + (math.sin(alpha) * math.cos(beta) * math.sin(gamma)),
		(math.cos(alpha) * math.cos(beta) * math.sin(gamma)) - (math.sin(alpha) * math.sin(beta) * math.cos(gamma))
	]])
	
	return quaternion