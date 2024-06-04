from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor

from motor_cortex.layers.guidance import GuidanceLayer
import torch
import einops
import torch.nn.functional as F
import numpy as np
import scipy

from diffuser_actor.utils.utils import (
    normalise_quat,
    compute_rotation_matrix_from_ortho6d
)
from typing import List, Tuple, Optional



class DiffuserActorGuided(DiffuserActor):
    def __init__(self, *args, **kwargs):
        super(DiffuserActorGuided, self).__init__(*args, **kwargs)

        if 'guidance_func_file' in kwargs:
            guidance_func_file =  kwargs['guidance_func_file']
        else:
            guidance_func_file = "guidance/guidance_func.py"
        
        self.guidance_factor = kwargs["guidance_factor"] if "guidance_factor" in kwargs else 1.0
        self.stochastic = kwargs["stochastic"] if "stochastic" in kwargs else False

        self.guidance_layer = GuidanceLayer(
            guidance_func_file=guidance_func_file,
            input_to_state=self.input_to_state,
            score_to_output=self.score_to_output,
            stochastic=self.stochastic
        )
    
    def set_guidance_func_file(self, guidance_func_file):
        print("!!!!!!!!!!!!!!!!Setting guidance_func_file: ", guidance_func_file)
        
        del self.guidance_layer
        self.guidance_func_file = guidance_func_file
        self.guidance_layer = GuidanceLayer(
            guidance_func_file=guidance_func_file,
            input_to_state=self.input_to_state,
            score_to_output=self.score_to_output,
            stochastic=self.stochastic
        )
        # check if the guidance_func was loaded
        if self.guidance_layer.guidance_func is None:
            print("!!!!!!!!!!!!!!!!!!!!!!Error: guidance_func is None")
            return True
        return False
    

    def compute_trajectories(self, *args, **kwargs):
        trajectory = super(DiffuserActorGuided, self).compute_trajectories(*args, **kwargs)
        # Do something else here
        print('Guided trajectories computed')
        print(trajectory)

        return trajectory   

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        # Condition on start-end pose
        B, nhist, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])

        return trajectory

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, 3+6+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            rgb_obs: (B, num_cameras, 3, H, W) in [0, 1]
            pcd_obs: (B, num_cameras, 3, H, W) in world coordinates
            instruction: (B, max_instruction_length, 512)
            curr_gripper: (B, nhist, output_dim)
        """

        # # get current gripper B value
        # B = curr_gripper.size(0)
        # print(B)

        if self._relative:
            pcd_obs, curr_gripper = self.convert2rel(pcd_obs, curr_gripper)
        if gt_trajectory is not None:
            gt_openess = gt_trajectory[..., 7:]
            gt_trajectory = gt_trajectory[..., :7]
        curr_gripper = curr_gripper[..., :7]

        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            
            # print input vars sizes for debugging
            print("init gt_trajectory size: ", gt_trajectory.size())
            print("init trajectory_mask size: ", trajectory_mask.size())
            print("init rgb_obs size: ", rgb_obs.size())
            print("init pcd_obs size: ", pcd_obs.size())
            print("init instruction size: ", instruction.size())
            print("init curr_gripper size: ", curr_gripper.size())


            # ---------------------- adequating model input for guidance ----------------------
            # generate n trajectories by repeating the first dimensions of each input variable n times

            trajectory_mask = trajectory_mask.repeat(self._num_samples, 1)
            rgb_obs = rgb_obs.repeat(self._num_samples, 1, 1, 1, 1)
            pcd_obs = pcd_obs.repeat(self._num_samples, 1, 1, 1, 1)
            instruction = instruction.repeat(self._num_samples, 1, 1)
            curr_gripper = curr_gripper.repeat(self._num_samples, 1, 1)
            
            # ---------------------- adequating model input for guidance ----------------------
                        
            # print input vars sizes for debugging
            print("-------------------------------------------- ")
            print("final gt_trajectory size: ", gt_trajectory.size())
            print("final trajectory_mask size: ", trajectory_mask.size())
            print("final rgb_obs size: ", rgb_obs.size())
            print("final pcd_obs size: ", pcd_obs.size())
            print("final instruction size: ", instruction.size())
            print("final curr_gripper size: ", curr_gripper.size())
            print()
            
            trajectories = self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper
            )

            # print traj sizes for debugging
            print("Trajectories size: ", trajectories.size())


            # ---------------------- guidance ----------------------
            guidance_output = self.guidance_layer.guide([trajectories])
            trajectoriy = guidance_output[0]
            # ---------------------- end of guidance ----------------------
            

            return trajectoriy
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[..., :3] = self.normalize_pos(curr_gripper[..., :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)

        # Prepare inputs
        fixed_inputs = self.encode_inputs(
            rgb_obs, pcd_obs, instruction, curr_gripper
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3],
            timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9],
            timesteps
        )
        noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition
        assert not cond_mask.any()

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps, fixed_inputs
        )

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            loss = (
                30 * F.l1_loss(trans, noise[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot, noise[..., 3:9], reduction='mean')
            )
            if torch.numel(gt_openess) > 0:
                openess = layer_pred[..., 9:]
                loss += F.binary_cross_entropy_with_logits(openess, gt_openess)
            total_loss = total_loss + loss
        return total_loss


     
    # ================== Guidance Wrapper functions ==================
    def input_to_state(self, model_output: List[torch.Tensor]):
        """
        function used by the guidance layer convert model output to robot state
        """



        return states, indices

    def score_to_output(self, model_output: List[torch.Tensor], guidance_score: torch.Tensor, indices):
        """
        function used by the guidance layer to convert model output to robot output
        """
        model_output_ = model_output

        return model_output_