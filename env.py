from typing import Callable, Optional, Union, List, Dict, Any
from collections import namedtuple
import os
from isaacgym import gymapi, gymtorch
import torch

from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply

def parse_kwarg(kwargs: dict, key: str, default_val: Any):
    return kwargs[key] if key in kwargs else default_val

class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[str]=None, local_pos: Optional[bool]=None,
        replay_speed: Optional[str]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.local_pos = local_pos
        self.parent_link = parent_link
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight

class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None
    CAMERA_POS= 0, -4.5, 2.0
    CAMERA_FOLLOWING = True

    def __init__(self,
        n_envs: int, fps: int=30, frameskip: int=2,
        episode_length: Optional[Union[Callable, int]] = 300,
        control_mode: str = "position",
        substeps: int = 2,
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,
        headless: bool = False,
        **kwargs
    ):
        self.viewer = None
        assert(control_mode in ["position", "torque", "free"])
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        self.substeps = substeps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(compute_device)
        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        self.headless = headless
        self.num_envs = n_envs
        self.compute_device = compute_device
        
        if graphics_device is None:
            graphics_device = compute_device
        self.graphics_device = graphics_device
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model
        if type(self.character_model) == str:
            self.character_model = [self.character_model]

        sim_params = self.setup_sim_params()
        
        # Configure headless mode
        if self.headless:
            sim_params.enable_cameras = True
            sim_params.headless = True
        
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        self.add_ground()
        self.envs, self.actors = self.create_envs(n_envs)
        n_actors_per_env = self.gym.get_actor_count(self.envs[0])
        self.actor_ids = torch.arange(n_actors_per_env * len(self.envs), dtype=torch.int32, device=self.device).view(len(self.envs), -1)
        controllable_actors = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            dof = self.gym.get_actor_dof_count(self.envs[0], i)
            if dof > 0: controllable_actors.append(i)
        self.actor_ids_having_dofs = \
            n_actors_per_env * torch.arange(len(self.envs), dtype=torch.int32, device=self.device).unsqueeze(-1) + \
            torch.tensor(controllable_actors, dtype=torch.int32, device=self.device).unsqueeze(-2)
        self.setup_action_normalizer()
        self.create_tensors()

        self.gym.prepare_sim(self.sim)

        self.root_tensor.fill_(0)
        self.gym.set_actor_root_state_tensor(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
        )
        self.joint_tensor.fill_(0)
        self.gym.set_dof_state_tensor(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
        )
        self.refresh_tensors()
        self.train()
        self.viewer_pause = False
        self.viewer_advance = False
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        base_pos = self.root_pos[tar_env].cpu()
        self.cam_target = gymapi.Vec3(*self.vector_up(0.89, [base_pos[0], base_pos[1], base_pos[2]]))

        self.simulation_step = 0
        self.lifetime = torch.zeros(len(self.envs), dtype=torch.int64, device=self.device)
        self.done = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime)

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1)
        self.rew_dim = self.reward().size(-1)

        for i in range(self.gym.get_actor_count(self.envs[0])):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            print("Links", sorted(rigid_body.items(), key=lambda x:x[1]), len(rigid_body))
            dof = self.gym.get_actor_dof_dict(self.envs[0], i)
            print("Joints", sorted(dof.items(), key=lambda x:x[1]), len(dof))

        # Setup headless camera if in headless mode
        if self.headless:
            self.setup_headless_camera()


    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

    def eval(self):
        self.training = False
        
    def train(self):
        self.training = True

    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector
    
    def setup_sim_params(self, physx_params=dict()):
        p = gymapi.SimParams()
        p.dt = self.step_time/self.frameskip
        p.substeps = self.substeps
        p.up_axis = gymapi.UP_AXIS_Z if self.UP_AXIS == 2 else gymapi.UP_AXIS_Y
        p.gravity = gymapi.Vec3(*self.vector_up(-9.81))
        p.num_client_threads = 0
        p.physx.num_threads = 4
        p.physx.solver_type = 1
        p.physx.num_subscenes = 4  # works only for CPU 
        p.physx.num_position_iterations = 4
        p.physx.num_velocity_iterations = 0
        p.physx.contact_offset = 0.01
        p.physx.rest_offset = 0.0
        p.physx.bounce_threshold_velocity = 0.2
        p.physx.max_depenetration_velocity = 10.0
        p.physx.default_buffer_size_multiplier = 5.0
        p.physx.max_gpu_contact_pairs = 8*1024*1024
        # p.physx.default_buffer_size_multiplier = 4
        # FIXME IsaacGym Pr4 will provide unreliable results when collecting from all substeps
        p.physx.contact_collection = \
            gymapi.ContactCollection(gymapi.ContactCollection.CC_LAST_SUBSTEP) 
        #gymapi.ContactCollection(gymapi.ContactCollection.CC_ALL_SUBSTEPS)
        for k, v in physx_params.items():
            setattr(p.physx, k, v)
        p.use_gpu_pipeline = True # force to enable GPU
        p.physx.use_gpu = True
        return p

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def add_actor(self, env, group):
        pass

    def create_envs(self, n: int, start_height=0.89, actuate_all_dofs=True, asset_options=dict()):
        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, actors = [], []
        env_spacing = 3

        actor_asset = []
        actuated_dof = []
        for character_model in self.character_model:
            asset_opt = gymapi.AssetOptions()
            asset_opt.angular_damping = 0.01
            asset_opt.max_angular_velocity = 100.0
            asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
            for k, v in asset_options.items():
                setattr(asset_opt, k, v)
            asset = self.gym.load_asset(self.sim,
                os.path.abspath(os.path.dirname(character_model)),
                os.path.basename(character_model),
                asset_opt)
            actor_asset.append(asset)
            if actuate_all_dofs:
                actuated_dof.append([i for i in range(self.gym.get_asset_dof_count(asset))])
            else:
                actuators = []
                for i in range(self.gym.get_asset_actuator_count(asset)):
                    name = self.gym.get_asset_actuator_joint_name(asset, i)
                    actuators.append(self.gym.find_asset_dof_index(asset, name))
                    if actuators[-1] == -1:
                        raise ValueError("Failed to find joint with name {}".format(name))
                actuated_dof.append(sorted(actuators) if len(actuators) else [])

        spacing_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        spacing_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        n_envs_per_row = int(n**0.5)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(start_height))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.actuated_dof = []
        for i in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            for aid, (asset, dofs) in enumerate(zip(actor_asset, actuated_dof)):
                actor = self.gym.create_actor(env, asset, start_pose, "actor{}_{}".format(i, aid), i, -1, 0)
                dof_prop = self.gym.get_asset_dof_properties(asset)
                for k in range(len(dof_prop)):
                    if k in dofs:
                        dof_prop[k]["driveMode"] = control_mode
                    else:
                        dof_prop[k]["stiffness"] = 0
                        dof_prop[k]["damping"] = 0
                self.gym.set_actor_dof_properties(env, actor, dof_prop)
                if i == n-1:
                    actors.append(actor)
                    self.actuated_dof.append(dofs)
            self.add_actor(env, i)
            envs.append(env)
        return envs, actors

    def setup_headless_camera(self, width=1280, height=720):
        """
        Sets up a camera for headless rendering
        """
        cam_props = gymapi.CameraProperties()
        cam_props.width = width
        cam_props.height = height
        cam_props.enable_tensors = True
        
        # Create the camera
        env_idx = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.headless_camera = self.gym.create_camera_sensor(self.envs[env_idx], cam_props)
        
        # Set initial camera position
        base_pos = self.root_pos[env_idx].cpu()
        cam_pos = gymapi.Vec3(
            base_pos[0]+self.camera_pos[0], 
            base_pos[1]+self.camera_pos[1], 
            base_pos[2]+self.camera_pos[2]
        )
        
        # Set camera target
        self.cam_target = gymapi.Vec3(*self.vector_up(0.89, [base_pos[0], base_pos[1], base_pos[2]]))
        
        self.gym.set_camera_location(self.headless_camera, self.envs[env_idx], cam_pos, self.cam_target)

    def render_headless(self, width=1280, height=720, camera_position=None, camera_target=None):
        """
        Renders the scene to an image without displaying a window.
        Returns the rendered RGB image.
        """
        if not hasattr(self, 'headless_camera'):
            self.setup_headless_camera(width, height)
        
        env_idx = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        
        # Update camera if needed
        if camera_position is not None or camera_target is not None or self.camera_following:
            self.update_camera_headless(camera_position, camera_target)
        
        # Perform rendering
        self.gym.step_graphics(self.sim)
        
        # Render the camera image
        self.gym.render_camera_sensor(self.sim, self.headless_camera)
        
        # Get the rendered image
        image = self.gym.get_camera_image(self.sim, self.envs[env_idx], self.headless_camera, gymapi.IMAGE_COLOR)
        
        # Reshape image to proper dimensions
        image = image.reshape(height, width, 4)  # RGBA format
        
        # Convert to RGB
        rgb_image = image[:, :, :3]
        
        return rgb_image

    def update_camera_headless(self, camera_position=None, camera_target=None):
        """
        Updates the camera position and target for headless rendering.
        """
        if not hasattr(self, 'headless_camera'):
            return
            
        env_idx = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        
        if camera_position is None and self.camera_following:
            # Update camera position based on root position
            base_pos = self.root_pos[env_idx].cpu()
            cam_trans = self.gym.get_camera_transform(self.sim, self.envs[env_idx], self.headless_camera)
            dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
            cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
            self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        elif camera_position is not None:
            cam_pos = gymapi.Vec3(*camera_position)
            if camera_target is not None:
                self.cam_target = gymapi.Vec3(*camera_target)
        else:
            return
            
        self.gym.set_camera_location(self.headless_camera, self.envs[env_idx], cam_pos, self.cam_target)

    def record_video(self, output_path, num_frames=1000, fps=30, width=1280, height=720):
        """
        Records a video of the simulation in headless mode.
        
        Args:
            output_path: Path to save the video
            num_frames: Number of frames to record
            fps: Frames per second
            width: Video width
            height: Video height
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            print("OpenCV is required for video recording. Please install with 'pip install opencv-python'")
            return
        
        if not self.headless:
            print("Warning: Recording video in non-headless mode. This may be slow.")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset environment
        self.reset()
        
        # Record frames
        for i in range(num_frames):
            # Get random actions (replace with your policy)
            actions = torch.zeros(len(self.envs), self.act_dim, device=self.device)
            
            # Step environment
            obs, rewards, dones, info = self.step(actions)
            
            # Reset done environments
            obs, info = self.reset_done()
            
            # Render frame
            frame = self.render_headless(width, height)
            
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            video.write(frame_bgr)
            
            if i % 100 == 0:
                print(f"Recorded {i}/{num_frames} frames")
        
        # Release video writer
        video.release()
        print(f"Video saved to {output_path}")

    def render(self):
        if self.headless:
            print("Warning: render() called in headless mode. Use render_headless() instead.")
            return
            
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(*self.vector_up(self.camera_pos[2], 
            [base_pos[0]+self.camera_pos[0], base_pos[1]+self.camera_pos[1], base_pos[2]+self.camera_pos[1]]))
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "TOGGLE_CAMERA_FOLLOWING")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "TOGGLE_PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "SINGLE_STEP_ADVANCE")
    
    def update_viewer(self):
        if self.headless or self.viewer is None:
            return
            
        self.gym.poll_viewer_events(self.viewer)
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "QUIT" and event.value > 0:
                exit()
            if event.action == "TOGGLE_CAMERA_FOLLOWING" and event.value > 0:
                self.camera_following = not self.camera_following
            if event.action == "TOGGLE_PAUSE" and event.value > 0:
                self.viewer_pause = not self.viewer_pause
            if event.action == "SINGLE_STEP_ADVANCE" and event.value > 0:
                self.viewer_advance = not self.viewer_advance
        if self.camera_following: self.update_camera()
        self.gym.step_graphics(self.sim)

    def update_camera(self):
        if self.headless or self.viewer is None:
            return
            
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
        base_pos = self.root_pos[tar_env].cpu()
        cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
        self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_tensors(self):
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(root_tensor)
        self.root_tensor = root_tensor.view(len(self.envs), -1, 13)

        num_links = self.gym.get_env_rigid_body_count(self.envs[0])
        link_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        link_tensor = gymtorch.wrap_tensor(link_tensor)
        self.link_tensor = link_tensor.view(len(self.envs), num_links, -1)

        num_dof = self.gym.get_env_dof_count(self.envs[0])
        joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_tensor = gymtorch.wrap_tensor(joint_tensor)
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)  # n_envs x n_dof x 2

        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_force_tensor = contact_force_tensor.view(len(self.envs), -1, 3)

        if self.actuated_dof.size(-1) == self.joint_tensor.size(1):
            self.action_tensor = None
        else:
            self.action_tensor = torch.zeros_like(self.joint_tensor[..., 0])

    def setup_action_normalizer(self):
        actuated_dof = []
        dof_cnts = 0
        action_lower, action_upper = [], []
        action_scale = []
        for i, dofs in zip(range(self.gym.get_actor_count(self.envs[0])), self.actuated_dof):
            actor = self.gym.get_actor_handle(self.envs[0], i)
            dof_prop = self.gym.get_actor_dof_properties(self.envs[0], actor)
            if len(dof_prop) < 1: continue
            if self.control_mode == "torque":
                action_lower.extend([-dof_prop["effort"][j] for j in dofs])
                action_upper.extend([dof_prop["effort"][j] for j in dofs])
                action_scale.extend([1]*len(dofs))
            else: # self.control_mode == "position":
                action_lower.extend([min(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_upper.extend([max(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_scale.extend([2]*len(dofs))
            for j in dofs:
                actuated_dof.append(dof_cnts+j)
            dof_cnts += len(dof_prop)
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale *= 0.5 * np.subtract(action_upper, action_lower)
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)
        self.actuated_dof = torch.tensor(actuated_dof, dtype=torch.int64, device=self.device)

    def process_actions(self, actions):
        a = actions*self.action_scale + self.action_offset
        if self.action_tensor is None:
            return a
        self.action_tensor[:, self.actuated_dof] = a
        return self.action_tensor

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False
        self.obs = None

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)
                if len(env_ids) == len(self.envs) or self.obs is None:
                    self.obs = self.observe()
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
    
    def reset_envs(self, env_ids):
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        self.root_tensor[env_ids] = ref_root_tensor
        self.link_tensor[env_ids] = ref_link_tensor
        if self.action_tensor is None:
            self.joint_tensor[env_ids] = ref_joint_tensor
        else:
            self.joint_tensor[env_ids.unsqueeze(-1), self.actuated_dof] = ref_joint_tensor

        actor_ids = self.actor_ids[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor),
            actor_ids, n_actor_ids
        )
        actor_ids = self.actor_ids_having_dofs[env_ids].flatten()
        n_actor_ids = len(actor_ids)
        actor_ids = gymtorch.unwrap_tensor(actor_ids)
        self.gym.set_dof_state_tensor_indexed(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor),
            actor_ids, n_actor_ids
        )

        self.lifetime[env_ids] = 0

    def do_simulation(self):
        for _ in range(self.frameskip):
            self.gym.simulate(self.sim)
        self.simulation_step += 1

    def step(self, actions):
        # Handle both GUI and headless modes
        if (not self.headless and not self.viewer_pause) or (self.headless) or (not self.headless and self.viewer_advance):
            self.apply_actions(actions)
            self.do_simulation()
            self.refresh_tensors()
            self.lifetime += 1
            if not self.headless and self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.viewer_advance = False

        # Update viewer in GUI mode
        if not self.headless and self.viewer is not None:
            self.update_viewer()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        # Or just step graphics in headless mode
        elif self.headless:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

        rewards = self.reward()
        terminate = self.termination_check()
        
        if self.headless or (not self.headless and not self.viewer_pause):
            overtime = self.overtime_check()
        else:
            overtime = None
            
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate
            
        self.info["terminate"] = terminate
        self.obs = self.observe()
        self.request_quit = False if self.viewer is None else self.gym.query_viewer_has_closed(self.viewer)
        return self.obs, rewards, self.done, self.info

    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        if self.control_mode == "position":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_position_target_tensor(self.sim, actions)
        elif self.control_mode == "torque":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        else:
            actions = torch.stack((actions, torch.zeros_like(actions)), -1)
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_