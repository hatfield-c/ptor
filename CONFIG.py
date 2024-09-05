import torch

possible_actions = {
	"help": "help", 
	"convert_static_env_mesh": "convert_static_env_mesh",
	"convert_rigid_drone_mesh": "convert_rigid_drone_mesh",
	"convert_visual_drone_mesh": "convert_visual_drone_mesh",
	"bake_env_data": "bake_env_data",
	"bake_rigidbody_data": "bake_rigidbody_data",
	"simulate": "simulate"
}

delta_time = 1 / 30
indices_per_meter = 10

gravity = torch.FloatTensor([[0, 0, -9.81]]).cuda()

static_env_mesh_path = "data/entities/portales/blender_meshes/sample_world.glb"
static_env_particles_path = "data/entities/portales/sample_world.pt"
static_env_baked_path = "data/entities/portales/sample_world.baked.pt"

rigid_drone_mesh_path = "data/entities/tau/blender_meshes/drone.glb"
rigid_drone_particles_path = "data/entities/tau/drone.pt"
rigid_drone_physics_path = "data/entities/tau/drone.physics.pt"

visual_drone_mesh_path = "data/entities/tau/blender_meshes/drone_visual.glb"
visual_drone_particles_path = "data/entities/tau/drone_visual.pt"