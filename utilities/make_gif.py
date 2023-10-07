import argparse
from PIL import Image
import os

def create_gif(input_dir, output_file):
    # Get a list of image file paths in the input directory
    image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    if not image_paths:
        print("No valid image files found in the directory.")
        return

    # Open the first image to get its size and create the GIF
    with Image.open(image_paths[0]) as first_image:
        # Create a list to hold frames
        frames = []

        # Iterate through each image, open it, and append it to the frames list
        for image_path in image_paths:
            with Image.open(image_path) as img:
                frames.append(img)

        # Save the frames as an animated GIF with an infinite loop (duration=0)
        frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=0, loop=0)
        print(f"Animated GIF saved to {output_file} with infinite loop")

def main():
    parser = argparse.ArgumentParser(description="Convert a directory of images to an animated GIF with an infinite loop.")
    parser.add_argument("input_directory", help="Input directory containing image files")
    parser.add_argument("output_file", help="Output animated GIF file")

    args = parser.parse_args()
    create_gif(args.input_directory, args.output_file)

if __name__ == "__main__":
    main()
