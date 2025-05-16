# create_video.py

import cv2
import os

def create_video(rgb_dir: str, output_video: str, frame_rate: int = 30):
    # Get the list of frames sorted by filename
    rgb_files = sorted(os.listdir(rgb_dir))
    
    if len(rgb_files) == 0:
        raise ValueError("No frames found in the specified directory.")

    # Load the first frame to determine the video size
    first_frame_path = os.path.join(rgb_dir, rgb_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        raise ValueError(f"Failed to load the first frame: {first_frame_path}")
    
    height, width, _ = first_frame.shape
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or other codecs for .avi or .mp4 formats
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    
    # Loop through all the frames and write them to the video
    for idx, rgb_file in enumerate(rgb_files):
        rgb_path = os.path.join(rgb_dir, rgb_file)
        rgb_frame = cv2.imread(rgb_path)

        if rgb_frame is None:
            print(f"Skipping frame {idx} - failed to load {rgb_path}")
            continue
        
        video_writer.write(rgb_frame)  # Add the frame to the video

        print(f"Processing frame {idx+1}/{len(rgb_files)}")

    # Release the video writer
    video_writer.release()
    print(f"Video created successfully: {output_video}")


if __name__ == "__main__":
    rgb_directory = "rgb/data"  # Update with your RGB frame folder
    output_file = "output_video.mp4"  # Output video file
    frame_rate = 30  # Set frame rate (frames per second)
    
    create_video(rgb_directory, output_file, frame_rate)
