import argparse
import os
import subprocess
from pydub import AudioSegment
import numpy as np
from scipy.signal import find_peaks
import json
import re


def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path
    ]
    subprocess.run(command)


def detect_hand_claps(audio_path, threshold=0.5, min_distance=10000):
    audio = AudioSegment.from_wav(audio_path)
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize the audio signal
    samples = samples / np.max(np.abs(samples))
    
    # Find peaks
    peaks, _ = find_peaks(samples, height=threshold, distance=min_distance)
    
    # Convert sample indices to time in seconds
    clap_times = [p / audio.frame_rate for p in peaks]
    
    return clap_times


def trim_video(video_path, start_time, end_time, output_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c", "copy",
        "-map", "0",
        "-map_metadata", "0",
        "-movflags", "use_metadata_tags",
        output_path
    ]
    subprocess.run(command)


def get_video_length(video_path):
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())


def extract_imu_data(video_path):
    # Run exiftool with -ee to extract per-frame metadata
    cmd = ['exiftool', '-ee2', '-n', '-G1', '-a', '-u', video_path]
    try:
        # Execute the command and capture the output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) #, text=True, check=True)
        output = result.stdout.decode('latin-1')
    except subprocess.CalledProcessError as e:
        print(f"Error executing ExifTool: {e.stderr}")
        return
    
    accel_data = []
    gyro_data = []

    # Split the output into lines for processing
    lines = output.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'Accelerometer' in line:
            match = re.search(r': ([\-\d\.]+) ([\-\d\.]+) ([\-\d\.]+)', line)
            if match:
                if i + 1 < len(lines):
                    if 'Time Stamp' in lines[i + 1]:
                        time_match = re.search(r': ([\-\d\.]+)', lines[i + 1])
                        accel_data.append([float(time_match.group(1)), float(match.group(1)), float(match.group(2)), float(match.group(3))])
            i += 1
        elif 'Magnetic Field' in line:
            match = re.search(r': ([\-\d\.]+) ([\-\d\.]+) ([\-\d\.]+)', line)
            if match:
                if i + 1 < len(lines):
                    if 'Time Stamp' in lines[i + 1]:
                        time_match = re.search(r': ([\-\d\.]+)', lines[i + 1])
                        gyro_data.append([float(time_match.group(1)), float(match.group(1)), float(match.group(2)), float(match.group(3))])
            i += 1
        else:
            i += 1

    return accel_data, gyro_data


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, default="/mnt/data/scene/", help="Directory containing all the 4 videos")
    args = argparser.parse_args()

    os.makedirs(os.path.join(args.input_dir, "data"), exist_ok=True)

    for i in range(2):
        if i == 0:
            video_path_1 = os.path.join(args.input_dir, "left_e1_er.MP4")
            video_path_2 = os.path.join(args.input_dir, "right_e2_er.MP4")
            audio_path_1 = os.path.join(args.input_dir, "left_e1_audio.wav")
            audio_path_2 = os.path.join(args.input_dir, "right_e2_audio.wav")
            output_video_path_1 = os.path.join(args.input_dir, "data/left_e1.mp4")
            output_video_path_2 = os.path.join(args.input_dir, "data/right_e2.mp4")

        if i == 1:
            video_path_1 = os.path.join(args.input_dir, "left_sfm_er.MP4")
            video_path_2 = os.path.join(args.input_dir, "right_sfm_er.MP4")
            audio_path_1 = os.path.join(args.input_dir, "left_sfm_audio.wav")
            audio_path_2 = os.path.join(args.input_dir, "right_sfm_audio.wav")
            output_video_path_1 = os.path.join(args.input_dir, "data/left_sfm.mp4")
            output_video_path_2 = os.path.join(args.input_dir, "data/right_sfm.mp4")

        extract_audio(video_path_1, audio_path_1)
        extract_audio(video_path_2, audio_path_2)

        clap_times_1 = detect_hand_claps(audio_path_1)
        clap_times_2 = detect_hand_claps(audio_path_2)

        if len(clap_times_1) < 2 or len(clap_times_2) < 2:
            raise ValueError("Not enough hand claps detected in one of the audios.")

        start_clap_1, end_clap_1 = clap_times_1[0], clap_times_1[-1]
        start_clap_2, end_clap_2 = clap_times_2[0], clap_times_2[-1]

        accel_data, gyro_data = extract_imu_data(video_path_1)
        accel_data_np = np.array(accel_data)
        
        accel_data_np[:, 0] = accel_data_np[:, 0] - accel_data_np[0, 0]
        start_index = (np.abs(accel_data_np[:, 0] - start_clap_1)).argmin()
        end_index = (np.abs(accel_data_np[:, 0] - end_clap_1)).argmin()

        gyro_data_np = np.array(gyro_data)
        gyro_data_np[:, 0] = gyro_data_np[:, 0] - gyro_data_np[0, 0]
        gyro_start_index = (np.abs(gyro_data_np[:, 0] - start_clap_1)).argmin()
        gyro_end_index = (np.abs(gyro_data_np[:, 0] - end_clap_1)).argmin()

        imu_data_json = {}
        imu_data_json['accel'] = accel_data_np[start_index:end_index, :].tolist()
        imu_data_json['gyro'] = gyro_data_np[gyro_start_index:gyro_end_index, :].tolist()

        with open(os.path.join(args.input_dir, "imu.json"), "w", encoding="UTF-8") as file:
            json.dump(imu_data_json, file, indent=4)
        
        trim_video(video_path_1, start_clap_1, end_clap_1, output_video_path_1)
        trim_video(video_path_2, start_clap_2, end_clap_2, output_video_path_2)

        if round(get_video_length(output_video_path_1)) != round(get_video_length(output_video_path_2)):
            print("clap_times_1: ", clap_times_1, "clap_times_2: ", clap_times_2)
            raise ValueError("The videos are not the same lenght.")

        os.remove(video_path_1)
        os.remove(video_path_2)
        os.remove(audio_path_1)
        os.remove(audio_path_2)

        print(f"Left video trimmed from {start_clap_1} to {end_clap_1}, length : {get_video_length(output_video_path_1)}")
        print(f"Right video trimmed from {start_clap_2} to {end_clap_2}, length : {get_video_length(output_video_path_2)}")
