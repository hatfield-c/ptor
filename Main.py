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

	if action == actions["playground"]:
		import PtorEngine
		
		engine = PtorEngine.PtorEngine()
		engine.InstantiateScenario()
		engine.Run()

	if action == actions["convert_static_mesh"]:
		import MeshConverter
		
		mesh_converter = MeshConverter.MeshConverter(CONFIG.static_mesh_path)
		mesh_converter.SaveAsVoxelParticles(CONFIG.static_particles_path, 0.1, 0)
		
	if action == actions["convert_rigid_mesh"]:
		import MeshConverter
		
		mesh_converter = MeshConverter.MeshConverter(CONFIG.rigid_mesh_path)
		mesh_converter.SaveAsObjectParticles(CONFIG.rigid_particles_path)
		
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

Main()
