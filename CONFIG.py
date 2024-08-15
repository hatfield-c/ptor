import torch

possible_actions = {
	"help": "help", 
	"convert_static_env_mesh": "convert_static_env_mesh",
	"convert_rigid_drone_mesh": "convert_rigid_drone_mesh",
	"convert_visual_drone_mesh": "convert_visual_drone_mesh",
	"simulate": "simulate"
}

delta_time = 1 / 30

static_env_mesh_path = "data/glb_meshes/sample_world.glb"
static_env_particles_path = "data/pt_particles/sample_world.pt"

rigid_drone_mesh_path = "data/glb_meshes/drone.glb"
rigid_drone_particles_path = "data/pt_particles/drone.pt"

visual_drone_mesh_path = "data/glb_meshes/drone_visual.glb"
visual_drone_particles_path = "data/pt_particles/drone_visual.pt"