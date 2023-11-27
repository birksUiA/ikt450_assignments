


# for each image in the folder, draw the timestamp on the image
# and save the image in a new folder

import cv2
from cv2.typing import MatLike
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
def draw_text(image: MatLike, text: str) -> MatLike:
    height, width, _ = image.shape

    # Calculate text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_text_width = int(width * 0.75)  # Adjust this factor for text scaling
    text_scale = max_text_width / 300  # Adjust the divisor for different scaling
    text_thickness = max(2, int(text_scale * 2))
    text_size = cv2.getTextSize(text, font, text_scale, text_thickness)[0]
    text_x = int((width - text_size[0]) / 2)
    text_y = int((height + text_size[1]) / 2)

    # Add text to the image
    cv2.putText(image, text, (text_x, text_y), font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return image

def draw_texts(path_in, path_out, text):
    # path_in: path to the folder containing the images
    # path_out: path to the folder containing the output images
    # path_timestamps: path to the folder containing the timestamps
    # timestamps: list of timestamps, each timestamp corresponding to an image

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    for f in tqdm(os.listdir(path_in)):
        img = cv2.imread(path_in + f)
        img = draw_text(img, text)
        cv2.imwrite(os.path.join(path_out, f), img)

def test_draw_timestamp():
    image = cv2.imread("data/evaluation/0_0.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    image_timestamp = draw_text(image, "2021-09-01 12:00:00") 
    image_rgb_timestamp = cv2.cvtColor(image_timestamp, cv2.COLOR_BGR2RGB)
    plt.figure(0)
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axes
    plt.figure(0)
    plt.imshow(image_rgb_timestamp)
    plt.axis('off')  # Hide axes
    plt.show()


def sample_imeages_wth_timestamps(path_images, path_images_timestamps):
    # plot 5 corresponding images and timestamps
    # path_images: path to the folder containing the images 
    # path_images_timestamps: path to the folder containing the images with timestamps
    images = os.listdir(path_images)
    images_timestamps = os.listdir(path_images_timestamps)
    for i in range(5):
        # plot images in 2 rows and 5 columns

        plt.subplot(2, 5, i+1)
        image = cv2.imread(path_images + images[i])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image_rgb)
        plt.axis('off')  # Hide axes
        
        plt.subplot(2, 5, i+6)
        image_timestamp = cv2.imread(path_images_timestamps + images_timestamps[i])
        image_timestamp_rgb = cv2.cvtColor(image_timestamp, cv2.COLOR_BGR2RGB)

        plt.imshow(image_timestamp_rgb)
        plt.axis('off')  # Hide axes
    plt.show()

def main():
    paths_real = {
            # name: [path_in, path_out]
            'training': ['data/training/', 'data/traning_timestamps/'],
            'validation': ['data/validation/', 'data/validation_timestamps/'],
            'evaluation': ['data/evaluation/', 'data/evaluation_timestamps/']
            }
    
    tests_paths = {
            # name: [path_in, path_out]
            'test': ['data/test/', 'data/test_timestamps/'],
            }
    paths = paths_real

    for key in paths.keys():
        print(f"Drawing timestamps for {key} data")
        draw_texts(paths[key][0], paths[key][1], "2021-09-01 12:00:00")

    for key in paths.keys():
        sample_imeages_wth_timestamps(paths[key][0], paths[key][1])
     
if __name__ == '__main__':
    main()
