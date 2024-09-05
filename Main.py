import argparse
import time

import CONFIG

def GetCliAction():
	print("")
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("-a", "--action", type = str, help = "what action to take. Must be one of the following: " + str(list(CONFIG.possible_actions.keys())))

	args = arg_parser.parse_args()

	action = args.action

	if action is None:
		print("	[Error]: You need to specify an action with the --action argument, i.e. --action [flag].")
		print("	For more information, try: --action help")
		exit()

	return action

def Main():
	start_time = time.time()

	actions = CONFIG.possible_actions
	action = GetCliAction()

	if action not in actions:
		action = actions["help"]

	if action == actions["simulate"]:
		import engine.PtorEngine as PtorEngine
		
		engine = PtorEngine.PtorEngine()
		engine.InstantiateScenario()
		engine.Run()

	if action == actions["convert_static_env_mesh"]:
		import engine.MeshConverter as MeshConverter
		
		mesh_converter = MeshConverter.MeshConverter(CONFIG.static_env_mesh_path)
		mesh_converter.SaveAsVoxelParticles(
			save_path = CONFIG.static_env_particles_path, 
			voxel_size = 0.1
		)
		
	if action == actions["convert_rigid_drone_mesh"]:
		import engine.MeshConverter as MeshConverter
		
		mesh_converter = MeshConverter.MeshConverter(CONFIG.rigid_drone_mesh_path)
		mesh_converter.SaveAsVoxelParticles(
			save_path = CONFIG.rigid_drone_particles_path, 
			voxel_size = 0.005,
			expected_mass = 2.806
		)
		
	if action == actions["convert_visual_drone_mesh"]:
		import engine.MeshConverter as MeshConverter
		
		mesh_converter = MeshConverter.MeshConverter(CONFIG.visual_drone_mesh_path)
		mesh_converter.SaveAsObjectParticles(save_path = CONFIG.visual_drone_particles_path)
		
	if action == actions["bake_env_data"]:
		import engine.ParticleBaker as ParticleBaker
		
		baker = ParticleBaker.ParticleBaker(CONFIG.rigid_drone_particles_path)
		baker.BakeEnvironmentData(CONFIG.static_env_baked_path)
		
	if action == actions["bake_rigidbody_data"]:
		import engine.ParticleBaker as ParticleBaker
		
		baker = ParticleBaker.ParticleBaker(CONFIG.rigid_drone_particles_path)
		baker.BakeRigidbodyData(CONFIG.rigid_drone_physics_path)
	
	if action == actions["train_position_estimator"]:
		import learning.PositionTrainer as PositionTrainer
		
		trainer = PositionTrainer.PositionTrainer()
		trainer.Learn()
	
	if action == "help":
		print("	[PTOR] Particle PyTorch Simulation")
		print("	PTOR is built on top of the PyTorch tensor library and simulates a the physics of a particle-based rigidbody.")

		print("	Usage:")
		print("		python Main.py --action [OPTIONS]")
		print("		python Main.py -a [OPTIONS]")

		print("	Options:")
		print("		help					Show this help menu.")
		print("     convert_static_mesh		Convert the static .obj mesh file into a .pt particle file.")
		print("     convert_rigid_mesh		Convert the rigid .obj mesh file into a .pt particle file.")
		print("		playground				Developer playground.")

	runtime = time.time() - start_time
	runtime = "{:.2f}".format(runtime)

	print("\n[" + action + "]: Operation complete")
	print("    Total runtime: " + runtime + " sec\n")


if __name__ == '__main__':
	Main()
