import os

import cv2
import numpy as np
import pandas as pd
import plotly.express as px

flash_cropped_path_well = 'cropped_well_exposed_test.mp4'
flash_cropped_path_fast = 'fast_exposure_test.mp4'
video_path_well = "small_flash_detecting_test_well_exposed.mp4"
video_path_fast = "small_flash_detecting_test_fast_exposed.mp4"
cap = cv2.VideoCapture(flash_cropped_path_fast)
if not cap.isOpened():
    print("Error: Could not open video.")
FPS = cap.get(cv2.CAP_PROP_FPS)
TIME_INTERVAL_WELL = [(13,17), (3*60+17, 3*60+21)]
TIME_INTERVAL_FAST = [(13,17), (3*60+17, 3*60+21)]

def max_instense_in_flashes(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    max_gray_list = np.array([])
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            time = frame_count / frame_rate
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # mask = gray>150
            # gray = gray * mask
            max_gray_in_frame = np.max(gray) / 255
            max_gray_list = np.append(max_gray_list, max_gray_in_frame)
            frame_count += 1
    finally:
        cap.release()

    return max_gray_list

max_instense_well = pd.DataFrame(max_instense_in_flashes(flash_cropped_path_well))
max_instense_fast = pd.DataFrame(max_instense_in_flashes(flash_cropped_path_fast))
max_instense_well_np = max_instense_well.to_numpy().reshape(-1)
max_instense_fast_np = max_instense_fast.to_numpy().reshape(-1)
well_gradient = pd.DataFrame(np.gradient(max_instense_well_np))
well_diff = pd.DataFrame(np.diff(max_instense_well_np, append=max_instense_well_np[-1]))
fast_gradient = pd.DataFrame(np.gradient(max_instense_fast_np))
fast_diff = pd.DataFrame(np.diff(max_instense_fast_np, append=max_instense_fast_np[-1]))

fig_mix = px.line()
fig_mix.add_scatter(x = max_instense_well.index, y =max_instense_well[max_instense_well.columns[0]], name = 'Well_Max_Gray')
fig_mix.add_scatter(x = max_instense_fast.index, y =max_instense_fast[max_instense_fast.columns[0]], name = 'Fast_Max_Gray')
#fig_mix.add_scatter(x = well_gradient.index, y =well_gradient[well_gradient.columns[0]], name="Well_Gradient")
#fig_mix.add_scatter(x = fast_gradient.index, y =fast_gradient[fast_gradient.columns[0]], name="Fast_Gradient")
fig_mix.add_scatter(x = well_diff.index, y =well_diff[well_diff.columns[0]], name="Well_Diff")
fig_mix.add_scatter(x = fast_diff.index, y =fast_diff[fast_diff.columns[0]], name="Fast_Diff")

fig_mix.show()

def find_signal_pair(signal_ori, signal_diff, fps, time_interval):
    '''
    time_interval: [(flash0_start_second, flashe_nd_second), (flash1_start_second, flash1_end_second)]. 
    For example:[(13, 15), (61, 62)]
    ''' 
    flash0_start_frame = time_interval[0][0] * fps
    flash0_end_frame = time_interval[0][1] * fps
    flash1_start_frame = time_interval[1][0] * fps
    flash1_end_frame = time_interval[1][1] * fps
    threshold_diff = np.percentile(signal_diff, 98)
    threshold_ori = 0.95

    # descending_signal_value = np.sort(signal_diff)[::-1]
    # descending_signal_index = np.argsort(signal_diff)[::-1]
    # max_signal_value = descending_signal_value[0]
    # min_signal_value = descending_signal_value[-1]
    signal_index_fall_down = np.where((signal_diff <= -threshold_diff) & (signal_ori >= threshold_ori))[0]
    signal_index_go_up = np.where((signal_diff >= threshold_diff) & (np.append(signal_ori[1:],[0]) >= threshold_ori))[0]+1
    
    def isSelected(np_array):
        return ((np_array >= flash0_start_frame) & (np_array <= flash0_end_frame)) | \
        ((np_array >= flash1_start_frame) & (np_array <= flash1_end_frame))
    
    selected_signal_index_fall_down = signal_index_fall_down[isSelected(signal_index_fall_down)]
    selected_signal_index_go_up = signal_index_go_up[isSelected(signal_index_go_up)]
    
    return {"Up_indices": selected_signal_index_go_up, "Down_indices" : selected_signal_index_fall_down}
well_pick_pairs = find_signal_pair(max_instense_well.to_numpy().reshape(-1), 
                                   well_diff.to_numpy().reshape(-1), 
                                   FPS, TIME_INTERVAL_WELL)
fast_pick_pairs = find_signal_pair(max_instense_fast.to_numpy().reshape(-1),
                                   fast_diff.to_numpy().reshape(-1),
                                   FPS, TIME_INTERVAL_FAST)
print(f"Well:{well_pick_pairs}")
print(f"Fast:{fast_pick_pairs}")

def synchronize_from_pick_pairs(well_pick_pairs, fast_pick_pairs):
    check_up_pairs = well_pick_pairs.get("Up_indices") - fast_pick_pairs.get("Up_indices")
    check_down_pairs = well_pick_pairs.get("Down_indices") - fast_pick_pairs.get("Down_indices")
    
    if check_up_pairs.var() == 0:
        print("Checked Successfully: Up signals pairs.")
        if check_down_pairs.var() == 0:
            print("Checked Successfully: down signals pairs.")
            print("Signals Matches Perfectly!")
            print(f"Well frame {check_up_pairs[0]} synchronizes with Fast frame 0")
    else:
        print("Signal MisMatch!")
    return check_up_pairs[0]

well_frames_shift = synchronize_from_pick_pairs(well_pick_pairs, fast_pick_pairs)

def extract_frames(video_path, output_folder, frames_range):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get video information
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Loop through each frame
    frame_index_in_range = 0
    for frame_number in range(frame_count):
        # Read the frame
        ret, frame = cap.read()
        
        # Save the frame as an image
        if ret and (frame_number in frames_range):
            frame_path = os.path.join(output_folder, f"frame_{frame_index_in_range}.png")
            cv2.imwrite(frame_path, frame)
            frame_index_in_range += 1
    # Release the video capture object
    cap.release()

def Extract_Imgs_from_synchronized_videos(video_path_well, video_path_fast, well_frames_shift):
    well_frame_count = int(cv2.VideoCapture(video_path_well).get(cv2.CAP_PROP_FRAME_COUNT))
    fast_frame_count = int(cv2.VideoCapture(video_path_fast).get(cv2.CAP_PROP_FRAME_COUNT))
    well_frames_range = range(well_frames_shift, well_frame_count)
    fast_frames_range = range(fast_frame_count)
    synchronized_frames_range = np.sort(np.intersect1d(np.array(well_frames_range)-well_frames_shift, np.array(fast_frames_range)))
    well_frames_range = range(synchronized_frames_range[-1]+1) + well_frames_shift
    fast_frames_range = range(synchronized_frames_range[-1]+1)
    
    extract_frames(video_path_well, "Well_images_synchronized",well_frames_range)
    extract_frames(video_path_fast, "Fast_images_synchronized",fast_frames_range)
    return synchronized_frames_range

Extract_Imgs_from_synchronized_videos(video_path_well, video_path_fast, well_frames_shift)
