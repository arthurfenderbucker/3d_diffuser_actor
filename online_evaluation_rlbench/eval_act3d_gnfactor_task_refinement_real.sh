exp=act3d_gnfactor


# tasks=(
#     push_buttons close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location put_groceries_in_cupboard put_item_in_drawer put_money_in_safe reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap
# )
# tasks=(
#     close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons
# )

# tasks=(
#     reach_and_drag
# )

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
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="front"
seed=0
# checkpoint=train_logs/act3d_gnfactor.pth
# checkpoint=train_logs/act3d_gnfactor_5ep.pth
checkpoint=train_logs/act3d_gnfactor_real_mouse_dragging.pth



num_ckpts=${#tasks[@]}

trap "echo; exit" INT

echo "ARGS: " ${filtered_args[@]}

for ((i=0; i<$num_ckpts; i++)); do

    CUDA_LAUNCH_BLOCKING=1 python online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --num_history 1 \
    --test_model act3d \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 0 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "quat_from_query" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations -1 \
    --max_tries $max_tries \
    --max_steps 20 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.08 ${filtered_args[@]};
    echo " ------------------------ DONE ---------------------------"
done

#     CUDA_LAUNCH_BLOCKING=1 python online_evaluation_rlbench/evaluate_policy.py \
#     --tasks ${tasks[$i]} \
#     --checkpoint $checkpoint \
#     --num_history 1 \
#     --test_model act3d \
#     --cameras $cameras \
#     --verbose $verbose \
#     --action_dim 8 \
#     --collision_checking 0 \
#     --predict_trajectory 0 \
#     --embedding_dim $embedding_dim \
#     --rotation_parametrization "quat_from_query" \
#     --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
#     --data_dir $data_dir \
#     --num_episodes $num_episodes \
#     --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
#     --use_instruction $use_instruction \
#     --instructions instructions/peract/instructions.pkl \
#     --variations {0..60} \
#     --max_tries $max_tries \
#     --max_steps 20 \
#     --seed $seed \
#     --gripper_loc_bounds_file $gripper_loc_bounds_file \
#     --gripper_loc_bounds_buffer 0.08 $@
    
