main_dir=Act3d_18Peract_20Demo_10GNFactortask

dataset=/data/datasets/peract/Peract_packaged/train_real
valset=/data/datasets/peract/Peract_packaged/val_real

lr=1e-4
num_ghost_points=1000
num_ghost_points_val=10000
B=12
C=120
ngpus=1
max_episodes_per_task=5 # number of episodes that you collect

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_keypose.py \
    --tasks mouse_dragging \
    --dataset $dataset \
    --valset $valset \
    --instructions instructions/peract/instructions_real.pkl \
    --gripper_loc_bounds tasks/real_tasks_location_bounds.json \
    --gripper_loc_bounds_buffer 0.08 \
    --num_workers 1 \
    --train_iters 600000 \
    --embedding_dim $C \
    --action_dim 8 \
    --use_instruction 1 \
    --weight_tying 1 \
    --gp_emb_tying 1 \
    --val_freq 4000 \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 14 \
    --cache_size 600 \
    --cache_size_val 0 \
    --variations {0..5} \
    --num_ghost_points $num_ghost_points\
    --num_ghost_points_val $num_ghost_points_val\
    --symmetric_rotation_loss 0 \
    --regress_position_offset 0 \
    --num_sampling_level 3 \
    --lr $lr\
    --position_loss_coeff 1 \
    --cameras front\
    --max_episodes_per_task $max_episodes_per_task \
    --run_log_dir act3d_multitask-C$C-B$B-lr$lr-real-eps$max_episodes_per_task

