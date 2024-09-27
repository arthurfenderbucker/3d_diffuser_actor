from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor

from motor_cortex.layers.guidance import GuidanceLayer
import torch
import torch.nn.functional as F

from diffuser_actor.utils.utils import (
    normalise_quat
)
from typing import List
from scipy.spatial.transform import Rotation as R


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

    def set_guidance_params(self, guidance_factor=0.0, stochastic=False):
        self.guidance_factor = guidance_factor
        self.stochastic = stochastic
        self.guidance_layer.stochastic = stochastic

    def set_guidance_func_file(self, guidance_func_file):
        
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
            return False
        return True
    

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
            if self.guidance_factor > 0.0:
                trajectory_mask = trajectory_mask.repeat(self.guidance_layer.num_samples, 1)
                rgb_obs = rgb_obs.repeat(self.guidance_layer.num_samples, 1, 1, 1, 1)
                pcd_obs = pcd_obs.repeat(self.guidance_layer.num_samples, 1, 1, 1, 1)
                instruction = instruction.repeat(self.guidance_layer.num_samples, 1, 1)
                curr_gripper = curr_gripper.repeat(self.guidance_layer.num_samples, 1, 1)
            
                        
            # print input vars sizes for debugging
            print("-------------------------------------------- ")
            print("final gt_trajectory size: \t", gt_trajectory.size(),"\t",gt_trajectory.device)
            print("final trajectory_mask size: \t", trajectory_mask.size(),"\t",trajectory_mask.device)
            print("final rgb_obs size: \t", rgb_obs.size(),"\t",rgb_obs.device)
            print("final pcd_obs size: \t", pcd_obs.size(),"\t",pcd_obs.device)
            print("final instruction size: \t", instruction.size(),"\t",instruction.device)
            print("final curr_gripper size: \t", curr_gripper.size(),"\t",curr_gripper.device)
            print()
            bs = gt_trajectory.size(0)

            # ---------------------- adequating model input for guidance ----------------------
            
            trajectories = self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper
            )

            # ---------------------- guidance ----------------------
            # print traj sizes for debugging
            print("batched trajectories size: ", trajectories.size())
            print("self.guidance_factor: ", self.guidance_factor)

            # reshape trajectories to have the sample in a dimention apart. (n_sample, bs, n, features)
            trajectories = trajectories.view(self.guidance_layer.num_samples, bs, trajectories.size(1), trajectories.size(2))
            print("Trajectories size: ", trajectories.size())

            if self.guidance_factor > 0.0:

                # sample around the outputs and get scores for each saple assuming a distribution of the outputs
                noisy_traj, noisy_traj_probs = self.guidance_layer.sample_around_outputs(trajectories, dims=[0,1,2,7], sigma=[0.1, 0.1, 0.1, 0.5])
                guidance_output = self.guidance_layer.guide([noisy_traj, noisy_traj_probs])
                trajectories = guidance_output[0]

                if self.guidance_layer.guidance_func is not None:
                    rotation = trajectories[:, -1, 3:7]
                    position = trajectories[:, -1, 0:3]
                    gripper = trajectories[:, -1, 7:]
                    r = R.from_quat(rotation.cpu().detach().numpy())
                    rotation_euler = r.as_euler('zyx', degrees=True)
                    output_state = position.cpu().detach().numpy().tolist()[0] + rotation_euler.tolist()[0] + gripper.cpu().detach().numpy().tolist()[0]
                    print("output_state: ", output_state)
                    out = self.guidance_layer.querie_guidance_func(output_state, update_vars_dict=True)
                    print(out)
                else:
                    print("guidance_func is None")
            # ---------------------- end of guidance ----------------------
        
            return trajectories
        
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
        n = self.guidance_layer.num_samples
        print("n: ", n)
        # guide only for the last position of the trajectory
        trajectories = model_output[0]
        print("Trajectories size: ", trajectories.size())

        # reshape the trajectories to get the last state of each trajectory
        end_states =  trajectories[:,:, -1, :]
        end_states = end_states.permute(1, 0, 2)

        print("End states size: ", end_states.size())
        
        if self._quaternion_format == 'xyzw':
            rot_quat = torch.concat([end_states[..., 4:7],end_states[..., 3:4]], dim=-1)
        else:
            rot_quat = end_states[..., 3:7]
        # convert quaternion to euler angles
        rot_quat_shape = rot_quat.size()
        print(" rot_quat_shape",rot_quat_shape)
        rot_quat = rot_quat.view(-1,rot_quat_shape[-1])
        r = R.from_quat(rot_quat.cpu().detach().numpy())
        rotation_euler = r.as_euler('zyx', degrees=True)
        rotation_euler = rotation_euler.reshape(rot_quat_shape[:-1] + (3,))
        print("rotation_euler size: ", rotation_euler.shape)
        
        # invert axis 0 and 1 to have the bs in the first axis
        end_states_shape= end_states.size()
        states = torch.zeros(end_states_shape[:-1] + (7,))
        states[:,:,0:3] = end_states[:,:,0:3] # guide only for the position
        states[:,:,3:6] = torch.tensor(rotation_euler).to(end_states.device)
        states[:,:,7:] = end_states[:,:,7:] # guide only for the gripper

        # states = torch.zeros_like(end_states)
        # states[:,:,0:3] = end_states[:,:,0:3] # guide only for the position
        indices = None
        return states, indices

    def score_to_output(self, model_output: List[torch.Tensor], guidance_score: torch.Tensor, indices):
        """
        function used by the guidance layer to convert model output to robot output
        """

        # states_std = self.gudance_layer.states_std
        trajectories = model_output[0]
        end_states_probs = model_output[1]
        print("Trajectories size: ", trajectories.size())
        print("end_states_probs size: ", end_states_probs.size())

        samples, bs, n_wp, features = trajectories.size()
        # end_states =  trajectories[:, -1, :].unsqueeze(1)
        # end_states = end_states.view(-1, self.guidance_layer.num_samples, end_states.shape[-1])

        guidance_mask = guidance_score.to(end_states_probs.device).permute(1,0).unsqueeze(-1)
        print("guidance_mask size: ", guidance_mask.size())
        # model_output_ = model_output.copy()

        combined_distribution = end_states_probs * (1.0-self.guidance_factor) + guidance_mask*self.guidance_factor

        if self.guidance_layer.stochastic:
            print("Applying stochastic")
            combined_distribution = self.guidance_layer.apply_stochastic(combined_distribution, 10)

        print("combined_distribution size: ", combined_distribution.size())
        # select the best trajectory based on the combined_distribution
        best_traj_idx = torch.argmax(combined_distribution, dim=0)
        print("Best trajectory index: ", best_traj_idx)
        
        bs_indices = torch.arange(bs).unsqueeze(1).expand(bs, n_wp)
        n_indices = torch.arange(n_wp).unsqueeze(0).expand(bs, n_wp)

        best_traj = trajectories[best_traj_idx, bs_indices, n_indices]
        print("Best trajectory size: ", best_traj.size())

        return [best_traj, combined_distribution]