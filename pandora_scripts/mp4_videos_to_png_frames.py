import argparse
import os
import subprocess

# TODO: limit number of frames... maybe 3 per second?

if __name__ == '__main__':
    # List of input video files
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, default="./data/lab_ground_floor/trimmed_videos")

    # get video files
    args = argparser.parse_args()
    videos = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".mp4")]
    if len(videos) == 0:
        raise Exception("No video files found in input directory.")

    for video in videos:
        # Get the video name without extension
        video_name = os.path.splitext(os.path.basename(video))[0]
        print(f"Extracting frames from {video}...")

        # Create a subfolder for the frames
        output_folder = os.path.join(args.input_dir, f'{video_name}')
        os.makedirs(output_folder, exist_ok=True)

        # Run FFmpeg command to convert video to PNG frames
        if video_name == 'left_sfm' or video_name == 'right_sfm':
            command = [
                "ffmpeg",
                "-i", video,
                "-vf", "fps=4.0",
                f"{output_folder}/output_%04d.png"
            ]
        else:
            command = [
                "ffmpeg",
                "-i", video,
                "-vf", "fps=15.0",
                f"{output_folder}/output_%04d.png"
            ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # only keep frames where mod 10 = 0
        for file in os.listdir(output_folder):
            if not file.endswith(".png"):
                continue
            frame_number = int(os.path.splitext(file)[0].split("_")[-1])
            if frame_number % 10 != 0:
                os.remove(os.path.join(output_folder, file))

    print("Frames extraction completed.")

