import os
import glob
import random
from typing import List, Dict, Any
from pathlib import Path
import json

import open3d
import traceback
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import einops

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from rlbench.demo import Demo
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from termcolor import colored

qt_path = os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
import cv2
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_path


try:
    import rospy
    import rospkg
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from cv_bridge import CvBridge

except ImportError:
    pass


from motor_cortex.common.guidance_wrapper import GuidanceWrapper, GuidanceArguments



ALL_RLBENCH_TASKS = [
    'basketball_in_hoop', 'beat_the_buzz', 'change_channel', 'change_clock', 'close_box',
    'close_door', 'close_drawer', 'close_fridge', 'close_grill', 'close_jar', 'close_laptop_lid',
    'close_microwave', 'hang_frame_on_hanger', 'insert_onto_square_peg', 'insert_usb_in_computer',
    'lamp_off', 'lamp_on', 'lift_numbered_block', 'light_bulb_in', 'meat_off_grill', 'meat_on_grill',
    'move_hanger', 'open_box', 'open_door', 'open_drawer', 'open_fridge', 'open_grill',
    'open_microwave', 'open_oven', 'open_window', 'open_wine_bottle', 'phone_on_base',
    'pick_and_lift', 'pick_and_lift_small', 'pick_up_cup', 'place_cups', 'place_hanger_on_rack',
    'place_shape_in_shape_sorter', 'place_wine_at_rack_location', 'play_jenga',
    'plug_charger_in_power_supply', 'press_switch', 'push_button', 'push_buttons', 'put_books_on_bookshelf',
    'put_groceries_in_cupboard', 'put_item_in_drawer', 'put_knife_on_chopping_board', 'put_money_in_safe',
    'put_rubbish_in_bin', 'put_umbrella_in_umbrella_stand', 'reach_and_drag', 'reach_target',
    'scoop_with_spatula', 'screw_nail', 'setup_checkers', 'slide_block_to_color_target',
    'slide_block_to_target', 'slide_cabinet_open_and_place_cups', 'stack_blocks', 'stack_cups',
    'stack_wine', 'straighten_rope', 'sweep_to_dustpan', 'sweep_to_dustpan_of_size', 'take_frame_off_hanger',
    'take_lid_off_saucepan', 'take_money_out_safe', 'take_plate_off_colored_dish_rack', 'take_shoes_out_of_box',
    'take_toilet_roll_off_stand', 'take_umbrella_out_of_umbrella_stand', 'take_usb_out_of_computer',
    'toilet_seat_down', 'toilet_seat_up', 'tower3', 'turn_oven_on', 'turn_tap', 'tv_on', 'unplug_charger',
    'water_plants', 'wipe_desk'
]
TASK_TO_ID = {task: i for i, task in enumerate(ALL_RLBENCH_TASKS)}


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


def load_episodes() -> Dict[str, Any]:
    with open(Path(__file__).parent.parent / "data_preprocessing/episodes.json") as fid:
        return json.load(fid)


class Mover:

    def __init__(self, task, disabled=False, max_tries=1):
        self._task = task
        self._last_action = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled
        self._raycast = False

    def __call__(self, action, collision_checking=False):
        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())
            criteria = (dist_pos < 5e-3,)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            action_collision = np.ones(action.shape[0]+1)
            action_collision[:-1] = action
            if collision_checking:
                action_collision[-1] = 0
            obs, reward, terminate = self._task.step(action_collision)

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate, images


class Actioner:

    def __init__(
        self,
        policy=None,
        instructions=None,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        action_dim=7,
        predict_trajectory=True
    ):

        self._policy = policy
        self._instructions = instructions
        self._apply_cameras = apply_cameras
        self._action_dim = action_dim
        self._predict_trajectory = predict_trajectory

        self._actions = {}
        self._instr = None
        self._task_str = None
        self._instr_text = None

        self._policy.eval()
        self.model = None
        self.tokenizer = None

    def load_encoding_model(self):


        import transformers
        self.model= transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    
    def encode_instruction(self, instr):

        if isinstance(instr, str):
            instr = [instr]
            
        tokens = self.tokenizer(instr, padding="max_length")["input_ids"]
        # lengths = [len(t) for t in tokens]
        # if any(l > 72 for l in lengths):
        #     raise RuntimeError(f"Too long instructions: {lengths}")

        tokens = torch.tensor(tokens)#.to(args.device)
        with torch.no_grad():
            pred = self.model(tokens).last_hidden_state
        instruction = pred.cpu()
        return instruction

    def load_episode(self, task_str, variation):
        self._task_str = task_str
        print(self._instructions[task_str][variation])
        print(len(self._instructions[task_str][variation]))
        print(self._instructions[task_str][variation].size())
        instructions = list(self._instructions[task_str][variation])
        # print("\n INSTRUCTIONS",self._instructions.keys())
        
        self._instr_idx = random.choice(range(len(instructions)))
        self._instr = instructions[self._instr_idx].unsqueeze(0)
        self._task_id = torch.tensor(TASK_TO_ID[task_str]).unsqueeze(0)
        self._actions = {}

    def get_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)

        action_ls = []
        trajectory_ls = []
        for i in range(len(key_frame)):
            obs = demo[key_frame[i]]
            action_np = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
            action = torch.from_numpy(action_np)
            action_ls.append(action.unsqueeze(0))

            trajectory_np = []
            for j in range(key_frame[i - 1] if i > 0 else 0, key_frame[i]):
                obs = demo[j]
                trajectory_np.append(np.concatenate([
                    obs.gripper_pose, [obs.gripper_open]
                ]))
            trajectory_ls.append(np.stack(trajectory_np))

        trajectory_mask_ls = [
            torch.zeros(1, key_frame[i] - (key_frame[i - 1] if i > 0 else 0)).bool()
            for i in range(len(key_frame))
        ]

        return action_ls, trajectory_ls, trajectory_mask_ls

    def predict(self, rgbs, pcds, gripper,
                interpolation_length=None):
        """
        Args:
            rgbs: (bs, num_hist, num_cameras, 3, H, W)
            pcds: (bs, num_hist, num_cameras, 3, H, W)
            gripper: (B, nhist, output_dim)
            interpolation_length: an integer

        Returns:
            {"action": torch.Tensor, "trajectory": torch.Tensor}
        """
        output = {"action": None, "trajectory": None}

        rgbs = rgbs / 2 + 0.5  # in [0, 1]

        # print(self._instr.shape)
        if self._instr is None:
            self._task_id = torch.tensor(TASK_TO_ID["push_buttons"]).unsqueeze(0)
            print(self._task_id.size())
            # check if model 
            try:
                if self.model is None:
                    print("loading models")
                    self.load_encoding_model()
                    print("encoding _instr_text: ", self._instr_text)
                    self._instr = self.encode_instruction(self._instr_text)
                else:
                    self._instr = self.encode_instruction(self._instr_text)
            except Exception as e:
                print(e)
                print("Could not encode instruction")


                self._instr = torch.zeros((1, 53, 512))
                

        self._instr = self._instr.to(rgbs.device)
        self._task_id = self._task_id.to(rgbs.device)

        # Predict trajectory
        if self._predict_trajectory:
            print('Predict Trajectory')
            fake_traj = torch.full(
                [1, interpolation_length - 1, gripper.shape[-1]], 0
            ).to(rgbs.device)
            traj_mask = torch.full(
                [1, interpolation_length - 1], False
            ).to(rgbs.device)
            output["trajectory"] = self._policy(
                fake_traj,
                traj_mask,
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[..., :7],
                run_inference=True
            )
        else:
            print('Predict Keypose')
            pred = self._policy(
                rgbs[:, -1],
                pcds[:, -1],
                self._instr,
                gripper[:, -1, :self._action_dim],
            )
            # Hackish, assume self._policy is an instance of Act3D
            output["action"] = self._policy.prepare_action(pred)

        return output

    @property
    def device(self):
        return next(self._policy.parameters()).device


def obs_to_attn(obs, camera):
    print("-===sv=======================================")
    print(obs.misc[f"{camera}_camera_extrinsics"])
    print(obs.misc[f"{camera}_camera_intrinsics"])

    print("-===sv=======================================")

    extrinsics_44 = torch.from_numpy(
        obs.misc[f"{camera}_camera_extrinsics"]
    ).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)


    intrinsics_33 = torch.from_numpy(
        obs.misc[f"{camera}_camera_intrinsics"]
    ).float()


    # T = torch.tensor([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).float()
    # rotate the extrinsics by 90 degrees around the z-axis
    # extrinsics_44[:, :] = extrinsics_44[:, :] @ T
    # extrinsics_44[:3, :3] = extrinsics_44[:3, :3] @ T[:3,:3]

    # extrinsics_44[0, 3] , extrinsics_44[1, 3]=extrinsics_44[1, 3],  extrinsics_44[0, 3]

    # flip x and y in the intrinsics
    # extrinsics_44[0, 3] , extrinsics_44[1, 3]= extrinsics_44[1, 3], extrinsics_44[0, 3]
    # intrinsics_33[0, 2] = -intrinsics_33[0, 2]
    # intrinsics_33[1, 2] = -intrinsics_33[1, 2]


    # intrinsics_33[0, 0],intrinsics_33[1, 1] = intrinsics_33[1, 1], intrinsics_33[0, 0]

    intrinsics_33[0,2] = 0
    intrinsics_33[1,2] = 0

    
    print(intrinsics_33)
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    print(u, v)
    # cv2.imshow("img", obs.front_rgb)
    import matplotlib.pyplot as plt
    plt.imshow(obs.front_rgb)
    w, h = obs.front_rgb.shape[:2]
    # plt.scatter(h-v,w-u)
    plt.scatter(u,v)

    # plt.show()
    plt.savefig("/root/motor_cortex/logs/test.png")
    plt.close()
    return u, v


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        image_size=(128, 128),
        image_stream_size=(128, 128),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist", "front"),
        fine_sampling_ball_diameter=None,
        collision_checking=False,
    ):


        # setup required inputs
        self.data_path = data_path
        self.apply_rgb = apply_rgb
        self.apply_depth = apply_depth
        self.apply_pc = apply_pc
        self.apply_cameras = apply_cameras
        self.fine_sampling_ball_diameter = fine_sampling_ball_diameter

        # setup RLBench environments
        obs_image_size = (max(image_size[0],image_stream_size[0]), max(image_size[1],image_stream_size[1]))
        self.obs_config = self.create_obs_config(
            obs_image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=Discrete()
        )
        
        # if self.ros_server:
        #     self.ros_setup()


        # ======== SETUP GUIDANCE =========
        
        guidance_args = GuidanceArguments().parse_args(known_only=True)

        # print(guidance_args)
        self.guidance_wrapper = GuidanceWrapper(guidance_args)
        self.rollouts_per_demo = self.guidance_wrapper.rollouts_per_demo

        if self.guidance_wrapper.pub_interval > 0:
            self.action_mode.arm_action_mode.set_callable_each_step(
                self.guidance_wrapper.get_obs_relay_func(self.get_obs_action))
        # ================================

        if self.guidance_wrapper.args.real_life:
            self.env = self.guidance_wrapper.get_real_life_env()
        else:
            self.env = Environment(
                self.action_mode, str(data_path), self.obs_config,
                headless=headless
            )
        self.image_size = image_size # image size that is rendered and transmitted
        self.obs_image_size = obs_image_size # models input size
    
    def resize_image(self, image):
        """
        Resize the image to the adequate model input size."""

        if self.obs_image_size != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        return image

    def get_obs_action(self, obs, extra_meta = {}):
        """
        Fetch the desired state and action based on the provided demo.
            :param obs: incoming obs
            :return: required observation and action list
        """

        meta = self.guidance_wrapper.get_obs_meta(obs)
        meta["robot_state"] = list(obs.gripper_pose)
        meta.update(extra_meta)

        # fetch state
        state_dict = {"rgb": [], "depth": [], "pc": []}
        
        for cam in self.apply_cameras:
            if self.apply_rgb:
                rgb = getattr(obs, "{}_rgb".format(cam))
                self.guidance_wrapper.transmit(rgb,f"{cam}_rgb", meta=meta)
                rgb = self.resize_image(rgb)
                state_dict["rgb"] += [rgb]

            if self.apply_depth:
                depth = getattr(obs, "{}_depth".format(cam))
                self.guidance_wrapper.transmit(depth,f"{cam}_depth", meta=meta)
                depth = self.resize_image(depth)
                state_dict["depth"] += [depth]

            if self.apply_pc:
                pc = getattr(obs, "{}_point_cloud".format(cam))
                self.guidance_wrapper.transmit(pc,f"{cam}_point_cloud", meta=meta)
                pc = self.resize_image(pc)
                state_dict["pc"] += [pc]

        error = self.guidance_wrapper.wait_redis_ak()
        # fetch action
        action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
        if error:
            print(colored("Error in guidance", "red"))
                
        return state_dict, torch.from_numpy(action).float()

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, gripper
        """
        state_dict, gripper = self.get_obs_action(obs)
        state = transform(state_dict, augmentation=False)
        state = einops.rearrange(
            state,
            "(m n ch) h w -> n m ch h w",
            ch=3,
            n=len(self.apply_cameras),
            m=2
        )
        rgb = state[:, 0].unsqueeze(0)  # 1, N, C, H, W
        pcd = state[:, 1].unsqueeze(0)  # 1, N, C, H, W
        gripper = gripper.unsqueeze(0)  # 1, D

        attns = torch.Tensor([])
        for cam in self.apply_cameras:
            u, v = obs_to_attn(obs, cam)# if not self.guidance_wrapper.args.real_life else (0, 0)
            attn = torch.zeros(1, 1, 1, self.image_size[0], self.image_size[1])
            if not (u < 0 or u > self.image_size[1] - 1 or v < 0 or v > self.image_size[0] - 1):
                attn[0, 0, 0, v, u] = 1
            attns = torch.cat([attns, attn], 1)
        rgb = torch.cat([rgb, attns], 2)

        return rgb, pcd, gripper

    def get_obs_action_from_demo(self, demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        key_frame.insert(0, 0)
        state_ls = []
        action_ls = []
        for f in key_frame:
            state, action = self.get_obs_action(demo._observations[f])
            state = transform(state, augmentation=False)
            state_ls.append(state.unsqueeze(0))
            action_ls.append(action.unsqueeze(0))
        return state_ls, action_ls

    def get_gripper_matrix_from_action(self, action):
        action = action.cpu().numpy()
        position = action[:3]
        quaternion = action[3:7]
        rotation = open3d.geometry.get_rotation_matrix_from_quaternion(
            np.array((quaternion[3], quaternion[0], quaternion[1], quaternion[2]))
        )
        gripper_matrix = np.eye(4)
        gripper_matrix[:3, :3] = rotation
        gripper_matrix[:3, 3] = position
        return gripper_matrix

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
            random_selection=False
        )
        return demos
    
    def evaluate_task_on_multiple_variations(
        self,
        task_str: str,
        max_steps: int,
        num_variations: int,  # -1 means all variations
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=100,
        num_history=1,
    ):
        self.guidance_wrapper.reset_seeds()
        self.env.launch()


        if self.guidance_wrapper.args.real_life:
            # task = self.guidance_wrapper.get_real_life_task(task_str)
            task = self.env.get_task()
            num_demos = 0
            print(" \n\nLOADED REAL TASK \n\n")
            task_variations = [0]
        else:
            task_type = task_file_to_task_class(task_str)
            task = self.env.get_task(task_type)

            task_variations = task.variation_count()

            if num_variations > 0:
                task_variations = np.minimum(num_variations, task_variations)
                task_variations = range(task_variations)
            else:
                task_variations = glob.glob(os.path.join(self.data_path, task_str, "variation*"))
                task_variations = [int(n.split('/')[-1].replace('variation', '')) for n in task_variations]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in task_variations:
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    num_demos=num_demos // len(task_variations) + 1,
                    actioner=actioner,
                    max_tries=max_tries,
                    verbose=verbose,
                    dense_interpolation=dense_interpolation,
                    interpolation_length=interpolation_length,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str: str,
        task: TaskEnvironment,
        max_steps: int,
        variation: int,
        num_demos: int,
        actioner: Actioner,
        max_tries: int = 1,
        verbose: bool = False,
        dense_interpolation=False,
        interpolation_length=50,
        num_history=0,
    ):
        device = actioner.device

        success_rate = 0
        num_valid_demos = 0
        total_reward = 0

        for demo_id in range(num_demos):
                
            if verbose:
                print(f"Starting demo {demo_id}")

            try:
                self.guidance_wrapper.reset_seeds()
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
                print(type(demo), demo)
                num_valid_demos += 1
            except Exception as e:
                print(colored(f"Couldnt load demo {demo_id} for {task_str} variation {variation}","red"))
                print(e)
                # print(e)
                # print()
                # traceback.print_exc()
                continue
            
            self.guidance_wrapper.reset_params()
            for self_improving_iteration in range(self.guidance_wrapper.args.guidance_iter, \
                                                  self.guidance_wrapper.args.guidance_iter + 1 + \
                                                  self.guidance_wrapper.self_improving_iterations ):
                
                print("=====================================")
                print(colored(f"STARTING Iteration {self_improving_iteration}","yellow"))
                if self_improving_iteration > 1:
                    
                    self.guidance_wrapper.reset_seeds()
                    demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
                    
                    self.guidance_wrapper.set_params_to_iteration(self_improving_iteration)

                success_rate, num_valid_demos, total_reward, max_reward, finished = self._evaluate_task_on_one_demo(demo,
                                                success_rate, num_valid_demos,
                                                total_reward, device, demo_id,
                                                task_str, task,
                                                max_steps,
                                                variation,
                                                num_demos,
                                                actioner,
                                                max_tries = max_tries,
                                                verbose = verbose,
                                                dense_interpolation=dense_interpolation,
                                                interpolation_length=interpolation_length,
                                                num_history=num_history,
                                                seed=0) # TODO parse seed
                if not finished:
                    print(colored(f"Demo {demo_id} failed on the iteration {self_improving_iteration}","red"))
                    break
                if max_reward > 0:
                    print(colored(f"Demo {demo_id} successful on the iteration {self_improving_iteration}","green"))
                    break # if the demo is successful, no need to continue
                        
        # Compensate for failed demos
        if num_valid_demos == 0:
            assert success_rate == 0
            valid = False
        else:
            valid = True

        
        return success_rate, valid, num_valid_demos

    def _evaluate_task_on_one_demo(self, demo,
                        success_rate, num_valid_demos,
                        total_reward, device, demo_id: int,
                        task_str: str, task: TaskEnvironment,
                        max_steps: int,
                        variation: int,
                        num_demos: int,
                        actioner: Actioner,
                        max_tries: int = 1,
                        verbose: bool = False,
                        dense_interpolation=False,
                        interpolation_length=50,
                        num_history=0,seed=0):
        
        rgbs = torch.Tensor([]).to(device)
        pcds = torch.Tensor([]).to(device)
        grippers = torch.Tensor([]).to(device)

        # descriptions, obs = task.reset()
        new_rollout = True
        max_reward=0.0
        finished = False
        for rollout in range(self.rollouts_per_demo):

            self.guidance_wrapper.set_experiment(task_str, variation, demo_id, rollout)
            
            from_best_iter =bool(self.guidance_wrapper.args.skip_successful)
            results, best_iter = self.guidance_wrapper.check_rollout_status(from_best_iter=from_best_iter)
            
            if self.guidance_wrapper.skip_existing and results is not None:
                print(results)
                if isinstance(results, dict):
                    print("checking for: ", f"{self.guidance_wrapper.guidance_factor}_{self.guidance_wrapper.guidance_iter}" )
                    it_res = results.get(f"{self.guidance_wrapper.guidance_factor}_{self.guidance_wrapper.guidance_iter}", None)
                    if it_res is not None and "max_reward" in it_res.keys() and "success_rate" in it_res.keys():
                        print(colored(f"SKIPPING ITER variation {variation} demo {demo_id} rollouts_sulfix {self.guidance_wrapper.rollouts_sulfix} rollout {rollout}","yellow"))
                        success_rate += it_res["success_rate"]
                        total_reward += it_res["max_reward"]
                        continue

                if best_iter == self.guidance_wrapper.guidance_iter or self.guidance_wrapper.guidance_iter == 1:
                    success_rate += results["success_rate"]
                    total_reward += results["max_reward"]
                    print(colored(f"SKIPPING variation {variation} demo {demo_id} rollouts_sulfix {self.guidance_wrapper.rollouts_sulfix} rollout {rollout}","yellow"))
                    continue
                else:
                    if self.guidance_wrapper.args.skip_successful and results["max_reward"] > 0:
                        print(colored(f"SKIPPING variation {variation} demo {demo_id} rollouts_sulfix {self.guidance_wrapper.rollouts_sulfix} rollout {rollout}","yellow"))
                        print(f"Best iteration {best_iter}")
                        continue
            
            # if self.guidance_wrapper.args.skip_successful:
            #     results = self.guidance_wrapper.check_last_iters_rollout_status()
            #     if results is not None:
            #         # success_rate += results["success_rate"]
            #         # total_reward += results["max_reward"]
            #         print(colored(f"Solved in previous iterations SKIPPING... variation {variation} demo {demo_id} rollouts_sulfix {self.guidance_wrapper.rollouts_sulfix} rollout {rollout} guidance_iter {self.guidance_wrapper.guidance_iter}","yellow"))
            #         continue

            descriptions, obs = task.reset_to_demo(demo)
            if self.guidance_wrapper.args.real_life:
                actioner._instr_text = descriptions[0]

                pass
            else:
                actioner.load_episode(task_str, variation)
                actioner._instr_text = descriptions[actioner._instr_idx]

            self.guidance_wrapper.set_task_description(actioner._instr_text)
            print(colored(actioner._instr_text,"blue"))
            
            # # task._task.set_initial_objects_in_scene()
            # print(task._task._initial_objs_in_scene)
            # for obj, objtype in task._task._initial_objs_in_scene:
            #     print(obj, objtype, str(type(obj)))
            #     if "Shape" in str(type(obj)):
            #         print(colored(obj.get_handle(),"yellow"))
            #         print(colored(f"{obj.get_name()}","red"))
            #         print(colored(f"{obj.get_position()}","red"))
            #         print(colored(f"{obj.get_orientation()}","red"))
            #         print(colored(f"color {obj.get_color()}","red"))

            #         # print(obj.get_object_name(obj.get_handle()))
            #         # print(dir(obj))


            if new_rollout:
                self.guidance_wrapper.trigger_code_generation()

            move = Mover(task, max_tries=max_tries)
            reward = 0.0
            max_reward = 0.0

            for step_id in range(max_steps):
                self.guidance_wrapper.set_step_id(step_id)
                # Fetch the current observation, and predict one action
                rgb, pcd, gripper = self.get_rgb_pcd_gripper_from_obs(obs)


                # ============== THE GUIDANCE CODE IS GENEREATED HERE ==============
                # ===        after sending the trigger and the fist obs.         === 
                # ==================================================================

                # wait for aknowledgement and update policy with guidance func if required 
                if step_id == 0 and new_rollout:
                    error = self.guidance_wrapper.add_guidance_to_policy(actioner._policy)
                    

                rgb = rgb.to(device)
                pcd = pcd.to(device)
                gripper = gripper.to(device)

                rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)

                # Prepare proprioception history
                rgbs_input = rgbs[:, -1:][:, :, :, :3]
                pcds_input = pcds[:, -1:]
                if num_history < 1:
                    gripper_input = grippers[:, -1]
                else:
                    gripper_input = grippers[:, -num_history:]
                    npad = num_history - gripper_input.shape[1]
                    gripper_input = F.pad(
                        gripper_input, (0, 0, npad, 0), mode='replicate'
                    )

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    gripper_input,
                    interpolation_length=interpolation_length
                )
                
                if verbose:
                    print(f"Step {step_id}")

                terminate = True

                # Update the observation based on the predicted action
                try:
                    # Execute entire predicted trajectory step by step
                    if output.get("trajectory", None) is not None:
                        trajectory = output["trajectory"][-1].cpu().numpy()
                        trajectory[:, -1] = trajectory[:, -1].round()

                        # execute
                        for action in tqdm(trajectory):
                            #try:
                            #    collision_checking = self._collision_checking(task_str, step_id)
                            #    obs, reward, terminate, _ = move(action_np, collision_checking=collision_checking)
                            #except:
                            #    terminate = True
                            #    pass
                            collision_checking = self._collision_checking(task_str, step_id)
                            obs, reward, terminate, _ = move(action, collision_checking=collision_checking)

                    # Or plan to reach next predicted keypoint
                    else:
                        # print("Plan with RRT")
                        action = output["action"]
                        action[..., -1] = torch.round(action[..., -1])
                        action = action[-1].detach().cpu().numpy()
                        if self.guidance_wrapper.args.pos_only:

                            # fixing the orientation and gripper
                            action[3:] = [0,0,0,1,0]
                            print(action[3:])

                        
                        # print(action)
                        collision_checking = self._collision_checking(task_str, step_id)
                        obs, reward, terminate, _ = move(action, collision_checking=collision_checking)

                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                    if terminate:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0

                # -------- LOGGING guidance --------
                self.guidance_wrapper.publish_guidance_info(actioner._policy)
                # ----------------------------------
                
            finished = True
            total_reward += max_reward
            if reward == 0:
                step_id += 1

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}", 
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", num_valid_demos,
            )
            
            new_rollout = False
            self.guidance_wrapper.log_success_rate(success_rate, max_reward, num_valid_demos)

        print("Rollouts completed")
        return success_rate, num_valid_demos, total_reward, max_reward, finished
    

    def _collision_checking(self, task_str, step_id):
        """Collision checking for planner."""
        # collision_checking = True
        collision_checking = False
        # if task_str == 'close_door':
        #     collision_checking = True
        # if task_str == 'open_fridge' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'open_oven' and step_id == 3:
        #     collision_checking = True
        # if task_str == 'hang_frame_on_hanger' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'take_frame_off_hanger' and step_id == 0:
        #     for i in range(300):
        #         self.env._scene.step()
        #     collision_checking = True
        # if task_str == 'put_books_on_bookshelf' and step_id == 0:
        #     collision_checking = True
        # if task_str == 'slide_cabinet_open_and_place_cups' and step_id == 0:
        #     collision_checking = True
        return collision_checking

    def verify_demos(
        self,
        task_str: str,
        variation: int,
        num_demos: int,
        max_tries: int = 1,
        verbose: bool = False,
    ):
        if verbose:
            print()
            print(f"{task_str}, variation {variation}, {num_demos} demos")

        self.env.launch()
        if self.guidance_wrapper.args.real_life:
            task = self.guidance_wrapper.get_real_life_task(task_str)
        else:
            task_type = task_file_to_task_class(task_str)
            task = self.env.get_task(task_type)
        task.set_variation(variation)  # type: ignore

        success_rate = 0.0
        invalid_demos = 0

        for demo_id in range(num_demos):
            if verbose:
                print(f"Starting demo {demo_id}")

            try:
                demo = self.get_demo(task_str, variation, episode_index=demo_id)[0]
            except:
                print(f"Invalid demo {demo_id} for {task_str} variation {variation}")
                print()
                traceback.print_exc()
                invalid_demos += 1

            task.reset_to_demo(demo)

            gt_keyframe_actions = []
            for f in keypoint_discovery(demo):
                obs = demo[f]
                action = np.concatenate([obs.gripper_pose, [obs.gripper_open]])
                gt_keyframe_actions.append(action)

            move = Mover(task, max_tries=max_tries)

            for step_id, action in enumerate(gt_keyframe_actions):
                if verbose:
                    print(f"Step {step_id}")

                try:
                    obs, reward, terminate, step_images = move(action)
                    if reward == 1:
                        success_rate += 1 / num_demos
                        break
                    if terminate and verbose:
                        print("The episode has terminated!")

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_type, demo, success_rate, e)
                    reward = 0
                    break

            if verbose:
                print(f"Finished demo {demo_id}, SR: {success_rate}")

        # Compensate for failed demos
        if (num_demos - invalid_demos) == 0:
            success_rate = 0.0
            valid = False
        else:
            success_rate = success_rate * num_demos / (num_demos - invalid_demos)
            valid = True

        self.env.shutdown()
        return success_rate, valid, invalid_demos


    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        unused_cams = CameraConfig()
        unused_cams.set_all(False)
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False, # mask=True,
            image_size=image_size,
            render_mode=RenderMode.OPENGL,
            **kwargs,
        )

        camera_names = apply_cameras
        kwargs = {}
        for n in camera_names:
            kwargs[n] = used_cams

        obs_config = ObservationConfig(
            front_camera=kwargs.get("front", unused_cams),
            left_shoulder_camera=kwargs.get("left_shoulder", unused_cams),
            right_shoulder_camera=kwargs.get("right_shoulder", unused_cams),
            wrist_camera=kwargs.get("wrist", unused_cams),
            overhead_camera=kwargs.get("overhead", unused_cams),
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True,
        )

        return obs_config

# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer, delta):
    next_is_not_final = i == (len(demo) - 2)
    # gripper_state_no_change = i < (len(demo) - 2) and (
    #     obs.gripper_open == demo[i + 1].gripper_open
    #     and obs.gripper_open == demo[i - 1].gripper_open
    #     and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    # )
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints


def transform(obs_dict, scale_size=(0.75, 1.25), augmentation=False):
    apply_depth = len(obs_dict.get("depth", [])) > 0
    apply_pc = len(obs_dict["pc"]) > 0
    num_cams = len(obs_dict["rgb"])

    obs_rgb = []
    obs_depth = []
    obs_pc = []
    for i in range(num_cams):
        rgb = torch.tensor(obs_dict["rgb"][i]).float().permute(2, 0, 1)
        depth = (
            torch.tensor(obs_dict["depth"][i]).float().permute(2, 0, 1)
            if apply_depth
            else None
        )
        pc = (
            torch.tensor(obs_dict["pc"][i]).float().permute(2, 0, 1) if apply_pc else None
        )

        if augmentation:
            raise NotImplementedError()  # Deprecated

        # normalise to [-1, 1]
        rgb = rgb / 255.0
        rgb = 2 * (rgb - 0.5)

        obs_rgb += [rgb.float()]
        if depth is not None:
            obs_depth += [depth.float()]
        if pc is not None:
            obs_pc += [pc.float()]
    obs = obs_rgb + obs_depth + obs_pc

    # print([o.shape for o in obs])
    return torch.cat(obs, dim=0)