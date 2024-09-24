from diffuser_actor.keypose_optimization.act3d import Act3D
from motor_cortex.layers.guidance import GuidanceLayer
import torch
import einops
import scipy

from typing import List
from diffuser_actor.utils.utils import (
    normalise_quat,
    compute_rotation_matrix_from_ortho6d
)

class Act3DGuided(Act3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'guidance_func_file' in kwargs:
            guidance_func_file =  kwargs['guidance_func_file']
        else:
            guidance_func_file = "guidance/guidance_func.py"
        
        # self.guidance_factor = kwargs["guidance_factor"] if "guidance_factor" in kwargs else 1.0
        # self.stochastic = kwargs["stochastic"] if "stochastic" in kwargs else False

        self.guidance_factor = 0.0
        self.stochastic = False

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


    def _predict_action(self,
                        ghost_pcd_mask, ghost_pcd, ghost_pcd_features, query_features, total_timesteps,
                        fine_ghost_pcd_offsets=None):
        """Compute the predicted action (position, rotation, opening) from the predicted mask."""
        # Select top-scoring ghost point
        top_idx = torch.max(ghost_pcd_mask, dim=-1).indices
        position = ghost_pcd[torch.arange(total_timesteps), :, top_idx]

        # Add an offset regressed from the ghost point's position to the predicted position
        if fine_ghost_pcd_offsets is not None:
            position = position + fine_ghost_pcd_offsets[torch.arange(total_timesteps), :, top_idx]

        # Predict rotation and gripper opening
        if self.rotation_parametrization in ["quat_from_top_ghost", "6D_from_top_ghost"]:
            ghost_pcd_features = einops.rearrange(ghost_pcd_features, "npts bt c -> bt npts c")
            features = ghost_pcd_features[torch.arange(total_timesteps), top_idx]
        elif self.rotation_parametrization in ["quat_from_query", "6D_from_query"]:
            features = query_features.squeeze(0)



        # ================== Guidance Infere predictor distribuiton ==================
        # pred, distribution = self.guidance_layer.infer_regression_model_distribution(self.gripper_state_predictor, features)
        # ============================================================================

        pred = self.gripper_state_predictor(features)

        print("rot dim: ",pred[:, :self.rotation_dim].size())
        print("gripper dim: ",pred[:, self.rotation_dim:].size())

        print("rot values: ",pred[:, :self.rotation_dim])
        print("gripper values: ",pred[:, self.rotation_dim:])

        if "quat" in self.rotation_parametrization:
            rotation = normalise_quat(pred[:, :self.rotation_dim])
        elif "6D" in self.rotation_parametrization:
            rotation = compute_rotation_matrix_from_ortho6d(pred[:, :self.rotation_dim])

        gripper = torch.sigmoid(pred[:, self.rotation_dim:])

        print("position: ", position)
        print("rotation: ", rotation)
        print("gripper: ", gripper)

        return position, rotation, gripper

    def forward(self, visible_rgb, visible_pcd, instruction, curr_gripper, gt_action=None):
        """
        Arguments:
            visible_rgb: (batch x history, num_cameras, 3, height, width) in [0, 1]
            visible_pcd: (batch x history, num_cameras, 3, height, width) in world coordinates
            curr_gripper: (batch x history, 8)
            instruction: (batch x history, max_instruction_length, 512)
            gt_action: (batch x history, 8) in world coordinates
        """
        total_timesteps, num_cameras, _, height, width = visible_rgb.shape
        device = visible_rgb.device
        if gt_action is not None:
            gt_position = gt_action[:, :3].unsqueeze(1).detach()
        else:
            gt_position = None
        curr_gripper = curr_gripper[:, :3]

        # Compute visual features at different scales and their positional embeddings
        visible_rgb_features_pyramid, visible_rgb_pos_pyramid, visible_pcd_pyramid = self._compute_visual_features(
            visible_rgb, visible_pcd, num_cameras)

        # Encode instruction
        if self.use_instruction:
            instruction_features = self.instruction_encoder(instruction)

            if self.ins_pos_emb:
                position = torch.arange(self._num_words)
                position = position.unsqueeze(0).to(instruction_features.device)

                pos_emb = self.instr_position_embedding(position)
                pos_emb = self.instr_position_norm(pos_emb)
                pos_emb = einops.repeat(pos_emb, "1 k d -> b k d", b=instruction_features.shape[0])

                instruction_features += pos_emb

            instruction_features = einops.rearrange(instruction_features, "bt l c -> l bt c")
            instruction_dummy_pos = torch.zeros(total_timesteps, instruction_features.shape[0], 3, device=device)
            instruction_dummy_pos = self.relative_pe_layer(instruction_dummy_pos)
        else:
            instruction_features = None
            instruction_dummy_pos = None

        # Compute current gripper position features and positional embeddings
        curr_gripper_pos = self.relative_pe_layer(curr_gripper.unsqueeze(1))
        curr_gripper_features = self.curr_gripper_embed.weight.repeat(total_timesteps, 1).unsqueeze(0)

        ghost_pcd_features_pyramid = []
        ghost_pcd_pyramid = []
        position_pyramid = []
        visible_rgb_mask_pyramid = []
        ghost_pcd_masks_pyramid = []

        for i in range(self.num_sampling_level):
            # Sample ghost points
            if i == 0:
                anchor = None
            else:
                anchor = gt_position if gt_position is not None else position_pyramid[-1]
            ghost_pcd_i = self._sample_ghost_points(total_timesteps, device, level=i, anchor=anchor)

            if i == 0:
                # Coarse RGB features
                visible_rgb_features_i = visible_rgb_features_pyramid[i]
                visible_rgb_pos_i = visible_rgb_pos_pyramid[i]
                ghost_pcd_context_features_i = einops.rearrange(
                    visible_rgb_features_i, "b ncam c h w -> (ncam h w) b c")
            else:
                # Local fine RGB features
                l2_pred_pos = ((position_pyramid[-1] - visible_pcd_pyramid[i]) ** 2).sum(-1).sqrt()
                indices = l2_pred_pos.topk(k=32 * 32 * num_cameras, dim=-1, largest=False).indices

                visible_rgb_features_i = einops.rearrange(
                    visible_rgb_features_pyramid[i], "b ncam c h w -> b (ncam h w) c")
                visible_rgb_features_i = torch.stack([
                    f[i] for (f, i) in zip(visible_rgb_features_i, indices)])
                visible_rgb_pos_i = torch.stack([
                    f[i] for (f, i) in zip(visible_rgb_pos_pyramid[i], indices)])
                ghost_pcd_context_features_i = einops.rearrange(
                    visible_rgb_features_i, "b npts c -> npts b c")

            # Compute ghost point features and their positional embeddings by attending to visual
            # features and current gripper position
            ghost_pcd_context_features_i = torch.cat(
                [ghost_pcd_context_features_i, curr_gripper_features], dim=0)
            ghost_pcd_context_pos_i = torch.cat([visible_rgb_pos_i, curr_gripper_pos], dim=1)
            if self.use_instruction:
                ghost_pcd_context_features_i = self.vis_ins_attn_pyramid[i](
                    query=ghost_pcd_context_features_i, value=instruction_features,
                    query_pos=None, value_pos=None
                )[-1]

                ghost_pcd_context_features_i = torch.cat(
                    [ghost_pcd_context_features_i, instruction_features], dim=0)
                ghost_pcd_context_pos_i = torch.cat(
                    [ghost_pcd_context_pos_i, instruction_dummy_pos], dim=1)
            (
                ghost_pcd_features_i,
                ghost_pcd_pos_i,
                ghost_pcd_to_visible_rgb_attn_i
            ) = self._compute_ghost_point_features(
                ghost_pcd_i, ghost_pcd_context_features_i, ghost_pcd_context_pos_i,
                total_timesteps, level=i
            )

            # Initialize query features
            if i == 0:
                query_features = self.query_embed.weight.unsqueeze(1).repeat(1, total_timesteps, 1)

            query_context_features_i = ghost_pcd_context_features_i
            query_context_pos_i = ghost_pcd_context_pos_i

            if i == 0:
                # Given the query is not localized yet, we don't use positional embeddings
                query_pos_i = None
                context_pos_i = None
            else:
                # Now that the query is localized, we use positional embeddings
                query_pos_i = self.relative_pe_layer(position_pyramid[-1])
                context_pos_i = query_context_pos_i

            # The query cross-attends to context features (visual features and the current gripper position)
            query_features = self._compute_query_features(
                query_features, query_context_features_i,
                query_pos_i, context_pos_i,
                level=i
            )



            # The query decodes a mask over ghost points (used to predict the gripper position) and over visual
            # features (for visualization only)
            ghost_pcd_masks_i, visible_rgb_mask_i = self._decode_mask(
                query_features,
                ghost_pcd_features_i, ghost_pcd_to_visible_rgb_attn_i,
                height, width, level=i
            )

            # ---------------------- guidance ----------------------
            # print("BEFORE ghost_pcd_masks_i[-1]: ", ghost_pcd_masks_i[-1].shape)
            guidance_output = self.guidance_layer.guide([ghost_pcd_masks_i[-1], ghost_pcd_i], level=i)
            ghost_pcd_masks_i[-1] = guidance_output[0]
            # print("AFTER ghost_pcd_masks_i[-1]: ", ghost_pcd_masks_i[-1].shape)

            query_features = query_features[-1]

            top_idx = torch.max(ghost_pcd_masks_i[-1], dim=-1).indices
            ghost_pcd_i = einops.rearrange(ghost_pcd_i, "b npts c -> b c npts")
            position_i = ghost_pcd_i[torch.arange(total_timesteps), :, top_idx].unsqueeze(1)

            ghost_pcd_pyramid.append(ghost_pcd_i)
            ghost_pcd_features_pyramid.append(ghost_pcd_features_i)
            position_pyramid.append(position_i)
            visible_rgb_mask_pyramid.append(visible_rgb_mask_i)
            ghost_pcd_masks_pyramid.append(ghost_pcd_masks_i)

        # Regress an offset from the ghost point's position to the predicted position
        if self.regress_position_offset:
            fine_ghost_pcd_offsets = self.ghost_point_offset_predictor(ghost_pcd_features_i)
            fine_ghost_pcd_offsets = einops.rearrange(fine_ghost_pcd_offsets, "npts b c -> b c npts")
        else:
            fine_ghost_pcd_offsets = None



        ghost_pcd = ghost_pcd_i
        ghost_pcd_masks = ghost_pcd_masks_i
        ghost_pcd_features = ghost_pcd_features_i

        # Predict the next gripper action (position, rotation, gripper opening)
        position, rotation, gripper = self._predict_action(
            ghost_pcd_masks[-1], ghost_pcd, ghost_pcd_features, query_features, total_timesteps,
            fine_ghost_pcd_offsets if self.regress_position_offset else None
        )
        # position = position_pyramid[-1].squeeze(1)

        # ---------------------- guidance: set previous_vars_dict ----------------------
        if self.guidance_layer.guidance_func is not None:
            r = scipy.spatial.transform.Rotation.from_quat(rotation.cpu().detach().numpy())
            rotation_euler = r.as_euler('zyx', degrees=True)
            output_state = position.cpu().detach().numpy().tolist()[0] + rotation_euler.tolist()[0] + gripper.cpu().detach().numpy().tolist()[0]
            out = self.guidance_layer.querie_guidance_func(output_state, update_vars_dict=True)
            print(out)
        else:
            print("guidance_func is None")

        # print(position)
        return {
            # Action
            "position": position,
            "rotation": rotation,
            "gripper": gripper,
            # Auxiliary outputs used to compute the loss or for visualization
            "position_pyramid": position_pyramid,
            "visible_rgb_mask_pyramid": visible_rgb_mask_pyramid,
            "ghost_pcd_masks_pyramid":  ghost_pcd_masks_pyramid,
            "ghost_pcd_pyramid": ghost_pcd_pyramid,
            "fine_ghost_pcd_offsets": fine_ghost_pcd_offsets if self.regress_position_offset else None,
            # Return intermediate results
            "visible_rgb_features_pyramid": visible_rgb_features_pyramid,
            "visible_pcd_pyramid": visible_pcd_pyramid,
            "query_features": query_features,
            "instruction_features": instruction_features,
            "instruction_dummy_pos": instruction_dummy_pos,
        }

    
    # ================== Guidance Wrapper functions ==================
    def input_to_state(self, model_output: List[torch.Tensor]):
        """
        function used by the guidance layer convert model output to robot state
        """
        ghost_pcd_mask_i, ghost_pcd_i = model_output

        indices = None
        states = torch.zeros(ghost_pcd_i.shape[0], ghost_pcd_i.shape[1] , 7)
        states[:,:,0:3] = ghost_pcd_i[indices]

        # states = torch.zeros(ghost_pcd_i.shape[0], ghost_pcd_i.shape[1] , 7)
        # #create a dummy state as a 3d grid
        # X = np.linspace(-1, 1, ghost_pcd_i.shape[1])
        # Y = np.linspace(-1, 1, ghost_pcd_i.shape[2])
        # Z = np.linspace(-1, 1, ghost_pcd_i.shape[3])
        # X, Y, Z = np.meshgrid(X, Y, Z)

        return states, indices

    def score_to_output(self, model_output: List[torch.Tensor], guidance_score: torch.Tensor, indices):
        
        ghost_pcd_mask_i, ghost_pcd_i = model_output

        guidance_mask = guidance_score.squeeze(-1).to(ghost_pcd_mask_i.device)

        #normalize ghost_pcd_mask_i as a probability distribution
        models_norm, _ = torch.max(ghost_pcd_mask_i[~torch.isnan(ghost_pcd_mask_i)], dim=-1, keepdim=True)
        models_distribution = ghost_pcd_mask_i / models_norm

        #normalize ghost_pcd_mask_i as a probability distribution
        # print("torch.max(guidance_mask, dim=-1, keepdim=True)", torch.max(guidance_mask, dim=-1, keepdim=True))
        # print("torch.min(guidance_mask, dim=-1, keepdim=True)", torch.min(guidance_mask, dim=-1, keepdim=True))
        # print("torch.mean(guidance_mask, dim=-1, keepdim=True)", torch.mean(guidance_mask, dim=-1, keepdim=True))


        # print("models_norm", models_norm)
        guidance_mask_max, _ = torch.max(guidance_mask, dim=-1, keepdim=True)
        if guidance_mask_max < 1e-6:
            guidance_mask_max = 1.0
            
        guidance_mask = guidance_mask / guidance_mask_max

        # copy the model_output to model_output_
        model_output_ = model_output.copy()
        model_output_[0] = models_distribution * (1.0-self.guidance_factor) + guidance_mask*self.guidance_factor
        # print("torch.max(model_output_[0].view(-1))", torch.max(model_output_[0].view(-1)))

        #scale model_output_[0] back
        model_output_[0] = model_output_[0] * models_norm
        # print("torch.max(model_output_[0].view(-1))", torch.max(model_output_[0].view(-1)))

        # print limits and std of ghost_pcd_mask_i
        
        # print("\n\nghost_pcd_mask_i: ", torch.max(ghost_pcd_mask_i), torch.min(ghost_pcd_mask_i), torch.mean(ghost_pcd_mask_i), torch.std(ghost_pcd_mask_i))
        # print("guidance_mask: ", torch.max(guidance_mask), torch.min(guidance_mask), torch.mean(guidance_mask), torch.std(guidance_mask))
        
        # scale_factor = torch.std(ghost_pcd_mask_i) #torch.max(ghost_pcd_mask_i) - torch.min(ghost_pcd_mask_i)
        # print("scale_factor: ", scale_factor)
        # print("scaled_guidance_mask: ", torch.max(guidance_mask), torch.min(guidance_mask), torch.std(guidance_mask))

        # guidance scaling the action probalities
        # guidance_mask = guidance_mask * self.guidance_factor
        # guidance_mask = guidance_mask + (1-torch.mean(guidance_mask))
        # model_output_ = model_output.copy()
        # model_output_[0] = ghost_pcd_mask_i * guidance_mask
        

        if self.guidance_layer.stochastic:
            model_output_[0] = self.guidance_layer.apply_stochastic(model_output_[0], 10)


        # self.guidance_layer.plot_guidance_summary(original_score=ghost_pcd_mask_i, guided_score = model_output[0], bins=100)
        return model_output_



    
