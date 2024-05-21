"""Online evaluation script on RLBench."""
import random
from typing import Tuple, Optional
from pathlib import Path
import json
import os

import torch
import numpy as np
import tap

from diffuser_actor.keypose_optimization.act3d_guided import Act3DGuided
from diffuser_actor.keypose_optimization.act3d import Act3D
from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from utils.common_utils import (
    load_instructions,
    get_gripper_loc_bounds,
    round_floats
)
from utils.utils_with_rlbench import RLBenchEnv, Actioner, load_episodes
import sys, signal # handling interruptions

class Arguments(tap.Tap):
    checkpoint: Path = ""
    seed: int = 2
    device: str = "cuda"
    num_episodes: int = 1
    headless: int = 0
    max_tries: int = 10
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = "instructions.pkl"
    variations: Tuple[int, ...] = (-1,)
    data_dir: Path = Path(__file__).parent / "demos"
    cameras: Tuple[str, ...] = ("left_shoulder", "right_shoulder", "wrist")
    image_size: str = "256,256"
    image_stream_size: str = "512,512"

    verbose: int = 0
    output_file: Path = Path(__file__).parent / "eval.json"
    max_steps: int = 25
    test_model: str = "3d_diffuser_actor"
    collision_checking: int = 0
    gripper_loc_bounds_file: str = "tasks/74_hiveformer_tasks_location_bounds.json"
    gripper_loc_bounds_buffer: float = 0.04
    single_task_gripper_loc_bounds: int = 0
    predict_trajectory: int = 1

    # Act3D model parameters
    num_query_cross_attn_layers: int = 2
    num_ghost_point_cross_attn_layers: int = 2
    num_ghost_points: int = 10000
    num_ghost_points_val: int = 10000
    weight_tying: int = 1
    gp_emb_tying: int = 1
    num_sampling_level: int = 3
    fine_sampling_ball_diameter: float = 0.16
    regress_position_offset: int = 0

    # 3D Diffuser Actor model parameters
    diffusion_timesteps: int = 100
    num_history: int = 3
    fps_subsampling_factor: int = 5
    lang_enhanced: int = 0
    dense_interpolation: int = 1
    interpolation_length: int = 2
    relative_action: int = 0

    # Shared model parameters
    action_dim: int = 8
    backbone: str = "clip"  # one of "resnet", "clip"
    embedding_dim: int = 120
    num_vis_ins_attn_layers: int = 2
    use_instruction: int = 1
    rotation_parametrization: str = '6D'

    # guidance parameters
    ros: int = 0
    redis: int = 0 # transmits data using redis server
    ak_topic: str = 'perception_ak'
    use_guidance: int = 0
    guidance_factor: float = 0.5
    generate_guidance_code: int = 0
    raw_policy: int = 0 #prevents loading checkpoint
    redis_pub_interval: int = 5 #transmition interval in frames, 0 means no transmission only key frames will be transmitted
    guidance_func_file: str = "/home/abucker/motor_cortex/benchmarks/3d_diffuser_actor/diffuser_actor/guidance/guidance_func.py"
    rollouts_per_demo: int = 1
    reuse_code: int = 1
    skip_existing: int = 0


def load_models(args):
    device = torch.device(args.device)

    print("Loading model from", args.checkpoint, flush=True)

    # Gripper workspace is the union of workspaces for all tasks
    if args.single_task_gripper_loc_bounds and len(args.tasks) == 1:
        task = args.tasks[0]
    else:
        task = None
    print('Gripper workspace')
    gripper_loc_bounds = get_gripper_loc_bounds(
        args.gripper_loc_bounds_file,
        task=task, buffer=args.gripper_loc_bounds_buffer,
    )

    if args.test_model == "3d_diffuser_actor":
        model = DiffuserActor(
            backbone=args.backbone,
            image_size=tuple(int(x) for x in args.image_size.split(",")),
            embedding_dim=args.embedding_dim,
            num_vis_ins_attn_layers=args.num_vis_ins_attn_layers,
            use_instruction=bool(args.use_instruction),
            fps_subsampling_factor=args.fps_subsampling_factor,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization=args.rotation_parametrization,
            diffusion_timesteps=args.diffusion_timesteps,
            nhist=args.num_history,
            relative=bool(args.relative_action),
            lang_enhanced=bool(args.lang_enhanced),
        )

    elif args.test_model == "act3d":
        if args.use_guidance:
            print("Using guidance")
            model = Act3DGuided(
                backbone=args.backbone,
                image_size=tuple(int(x) for x in args.image_size.split(",")),
                embedding_dim=args.embedding_dim,
                num_ghost_point_cross_attn_layers=(
                    args.num_ghost_point_cross_attn_layers),
                num_query_cross_attn_layers=(
                    args.num_query_cross_attn_layers),
                num_vis_ins_attn_layers=(
                    args.num_vis_ins_attn_layers),
                rotation_parametrization=args.rotation_parametrization,
                gripper_loc_bounds=gripper_loc_bounds,
                num_ghost_points=args.num_ghost_points,
                num_ghost_points_val=args.num_ghost_points_val,
                weight_tying=bool(args.weight_tying),
                gp_emb_tying=bool(args.gp_emb_tying),
                num_sampling_level=args.num_sampling_level,
                fine_sampling_ball_diameter=(
                    args.fine_sampling_ball_diameter),
                regress_position_offset=bool(
                    args.regress_position_offset),
                use_instruction=bool(args.use_instruction),
                guidance_factor=args.guidance_factor,
                stochastic=True if args.rollouts_per_demo > 1 else False,
            ).to(device)
        else:
            model = Act3D(
                backbone=args.backbone,
                image_size=tuple(int(x) for x in args.image_size.split(",")),
                embedding_dim=args.embedding_dim,
                num_ghost_point_cross_attn_layers=(
                    args.num_ghost_point_cross_attn_layers),
                num_query_cross_attn_layers=(
                    args.num_query_cross_attn_layers),
                num_vis_ins_attn_layers=(
                    args.num_vis_ins_attn_layers),
                rotation_parametrization=args.rotation_parametrization,
                gripper_loc_bounds=gripper_loc_bounds,
                num_ghost_points=args.num_ghost_points,
                num_ghost_points_val=args.num_ghost_points_val,
                weight_tying=bool(args.weight_tying),
                gp_emb_tying=bool(args.gp_emb_tying),
                num_sampling_level=args.num_sampling_level,
                fine_sampling_ball_diameter=(
                    args.fine_sampling_ball_diameter),
                regress_position_offset=bool(
                    args.regress_position_offset),
                use_instruction=bool(args.use_instruction)
            ).to(device)
    elif args.test_model == "peract":
        print("peract")
    else:
        raise NotImplementedError

    # Load model weights
    model_dict = torch.load(args.checkpoint, map_location="cpu")
    model_dict_weight = {}
    for key in model_dict["weight"]:
        _key = key[7:]
        model_dict_weight[_key] = model_dict["weight"][key]
    model.load_state_dict(model_dict_weight)
    model.eval()

    return model


if __name__ == "__main__":
    # Arguments


    def signal_handler(signal, frame):
        print("\nprogram exiting gracefully")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    args = Arguments().parse_args()
    args.cameras = tuple(x for y in args.cameras for x in y.split(","))
    print("Arguments:")
    print(args)
    print("-" * 100)
    # Save results here
    if args.use_guidance:
        output_path = str(args.output_file).split('/')
        output_file = os.path.dirname(os.path.dirname(str(args.output_file))) + f"_guided-{args.use_guidance}_f-{args.guidance_factor}/"+"/".join(output_path[-2:])
    else:
        output_file = args.output_file
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print("output path:", os.path.dirname(output_file))

    # # # Ros setup
    if args.ros:
        import logging
        logging._srcfile = None
        import rospy
        print("Initializing ROS node")
        rospy.init_node("rlbench_online_evaluation")
        rospy.loginfo("ROS node initialized")

    r = None
    if args.redis:
        import redis
        r = redis.StrictRedis(host='localhost', port=6379, db=0)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load models
    model = load_models(args)

    # Load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        image_size=[int(x) for x in args.image_size.split(",")],
        image_stream_size=[int(x) for x in args.image_stream_size.split(",")],
        apply_rgb=True,
        apply_pc=True,
        headless=bool(args.headless),
        apply_cameras=args.cameras,
        collision_checking=bool(args.collision_checking),
        server_args = {"ros_server": args.ros,
                       "redis_server": r,
                       "ak_topic": args.ak_topic,
                       "generate_guidance_code":args.generate_guidance_code,
                       "use_guidance": args.use_guidance,
                       "redis_pub_interval": args.redis_pub_interval,
                       "rollouts_per_demo": args.rollouts_per_demo,
                       "guidance_factor": args.guidance_factor,
                       "reuse_code": args.reuse_code,
                       "skip_existing": args.skip_existing} if args.redis else None,
    )

    instruction = load_instructions(args.instructions)
    if instruction is None:
        raise NotImplementedError()

    actioner = Actioner(
        policy=model,
        instructions=instruction,
        apply_cameras=args.cameras,
        action_dim=args.action_dim,
        predict_trajectory=bool(args.predict_trajectory)
    )
    max_eps_dict = load_episodes()["max_episode_length"]
    task_success_rates = {}

    for task_str in args.tasks:
        var_success_rates = env.evaluate_task_on_multiple_variations(
            task_str,
            max_steps=(
                max_eps_dict[task_str] if args.max_steps == -1
                else args.max_steps
            ),
            num_variations=args.variations[-1] + 1,
            num_demos=args.num_episodes,
            actioner=actioner,
            max_tries=args.max_tries,
            dense_interpolation=bool(args.dense_interpolation),
            interpolation_length=args.interpolation_length,
            verbose=bool(args.verbose),
            num_history=args.num_history
        )
        print()
        print(
            f"{task_str} variation success rates:",
            round_floats(var_success_rates)
        )
        print(
            f"{task_str} mean success rate:",
            round_floats(var_success_rates["mean"])
        )

        task_success_rates[task_str] = var_success_rates
        with open(output_file, "w") as f:
            json.dump(round_floats(task_success_rates), f, indent=4)
