from motor_cortex.common.perception_functions import get_position, get_size, get_orientation
import numpy

def guidance(state):
    # Constants for scoring
    NEAR_THRESHOLD = 0.05  # 5 cm is considered near
    ALIGNMENT_THRESHOLD = 0.01  # 1 cm tolerance for alignment
    ROTATION_THRESHOLD = 0.1  # 0.1 radian rotation tolerance
    GRIPPER_OPEN_THRESHOLD = 0.02  # 2 cm considered as gripper open enough to grasp the lid
    ORIENTATION_MATCH_THRESHOLD = 0.1  # 0.1 radian for orientation match
    
    # Get the current state of the robot's end-effector
    robot_x, robot_y, robot_z, robot_rx, robot_ry, robot_rz, gripper = state
    
    # Get the positions and orientations of the jar and the lid
    jar_pos = get_position('jar')
    lid_pos = get_position('lid')
    jar_ori = get_orientation('jar')
    lid_ori = get_orientation('lid')
    
    # Get the sizes of the jar and the lid
    jar_size = get_size('jar')
    lid_size = get_size('lid')
    
    # Calculate distance and alignment scores
    distance_to_lid = numpy.linalg.norm(numpy.array([robot_x, robot_y, robot_z]) - numpy.array(lid_pos))
    distance_score = max(0, (NEAR_THRESHOLD - distance_to_lid) / NEAR_THRESHOLD)
    
    alignment_score = max(0, (ALIGNMENT_THRESHOLD - numpy.linalg.norm(numpy.array([robot_rx, robot_ry, robot_rz]) - numpy.array(lid_ori))) / ALIGNMENT_THRESHOLD)
    
    # Check if the gripper is open enough to grasp the lid
    gripper_open_score = 1 if gripper >= GRIPPER_OPEN_THRESHOLD else 0
    
    # Check if the robot is in position to close the jar (lid above the jar and aligned)
    jar_lid_aligned = numpy.linalg.norm(numpy.array(jar_pos) - numpy.array(lid_pos)) < ALIGNMENT_THRESHOLD
    lid_above_jar = (jar_pos[2] + jar_size[2]/2 + lid_size[2]/2) <= robot_z <= (jar_pos[2] + jar_size[2]/2 + lid_size[2] * 1.5)
    position_to_close_score = 1 if jar_lid_aligned and lid_above_jar else 0
    
    # Check if the lid orientation matches the jar orientation for closure
    orientation_match_score = max(0, (ORIENTATION_MATCH_THRESHOLD - numpy.linalg.norm(numpy.array(jar_ori) - numpy.array(lid_ori))) / ORIENTATION_MATCH_THRESHOLD)
    
    # Calculate rotation score if the robot is in position to close the jar
    if position_to_close_score and orientation_match_score > 0:
        # Assuming the robot needs to rotate the lid to close it, we would have a target rotation
        # For simplicity, let's assume we just need to check the rotation around the z-axis
        target_rotation = jar_ori[2]  # Assuming jar orientation is the target for the lid
        rotation_score = max(0, (ROTATION_THRESHOLD - abs(robot_rz - target_rotation)) / ROTATION_THRESHOLD)
    else:
        rotation_score = 0
    
    # Weigh the scores based on the current stage of the task
    if distance_to_lid > NEAR_THRESHOLD:
        overall_score = distance_score * 0.7 + alignment_score * 0.2 + gripper_open_score * 0.1
    elif orientation_match_score > 0:
        overall_score = position_to_close_score * 0.4 + rotation_score * 0.6
    else:
        overall_score = orientation_match_score
    
    return overall_score

# Example robot state (you would get this from the actual robot sensors)
# robot_state = (x, y, z, rotation_x, rotation_y, rotation_z, gripper)

# Example call to the guidance function
# score = guidance(robot_state)