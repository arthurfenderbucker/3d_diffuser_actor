exp=3d_diffuser_actor_gnfactor_ep5

# Initialize an empty array to hold the filtered arguments
filtered_args=()

# A flag to indicate whether the next argument(s) are tasks
skip_tasks=false

# default tasks
tasks=(
    close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons
)
echo "$@"
# Iterate over all the arguments passed to the script
for arg in "$@"; do
    if [ "$arg" == "--tasks" ]; then
        skip_tasks=true  # Start skipping the tasks
        tasks=()  # re initialize an empty array to hold the tasks
    elif [ "$skip_tasks" == true ]; then
        # Check if the argument starts with '--', indicating the end of the task list
        if [[ $arg == --* ]]; then
            skip_tasks=false  # End of task list
            filtered_args+=("$arg")  # Add the argument to the filtered list
        else
            tasks+=("$arg")  # Add the task to the list
        fi
        # Otherwise, continue skipping task arguments
    else
        filtered_args+=("$arg")  # Add non-task arguments to the filtered list
    fi
done

data_dir=./data/peract/raw/test/
num_episodes=100
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=0
checkpoint=train_logs/diffuser_actor_gnfactor_ep5.pth
quaternion_format=xyzw #for policies trained locally

# checkpoint=train_logs/diffuser_actor_gnfactor.pth
# quaternion_format=wxyz # for the checkpoint provided by the authors


trap "echo; exit" INT
echo "ARGS: " ${filtered_args[@]}
num_ckpts=${#tasks[@]}


for ((i=0; i<$num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1 python online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --diffusion_timesteps 100 \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --relative_action $relative_action \
    --num_history 3 \
    --test_model 3d_diffuser_actor \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "6D" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations -1 \
    --max_tries $max_tries \
    --max_steps 10 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.08 \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation 1 ${filtered_args[@]};
done

