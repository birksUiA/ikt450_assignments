import dataloader
import helper as hp
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def main():
    # stop
    image_size=(244, 244)
    data_test = dataloader.load_data_pair_no_label(
        path_x=dataloader.tests_paths["test"][0],
        path_y=dataloader.tests_paths["test"][1],
        image_size=image_size,
    )
    data_training = dataloader.load_data_pair_no_label(
        path_x=dataloader.paths_real["training"][0],
        path_y=dataloader.paths_real["training"][1],
        image_size=image_size,
    )
    data_validation = dataloader.load_data_pair_no_label(
        path_x=dataloader.paths_real["validation"][0],
        path_y=dataloader.paths_real["validation"][1],
        image_size=image_size,
    )
    data_evaluation = dataloader.load_data_pair_no_label(
        path_x=dataloader.paths_real["evaluation"][0],
        path_y=dataloader.paths_real["evaluation"][1],
        image_size=image_size,
    )
    # Plot some images
    image: tf.data.Dataset = data_test.take(1)
    
    # subfigure 
    for image_x, image_y in image:

        print(image_x.shape)
        print(image_y.shape)
        print(f"{np.max(image_x[0])} {np.min(image_x[0])}")
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image_x) 
        plt.subplot(1, 2, 2)
        plt.imshow(image_y)
        plt.show()

if __name__ == "__main__":
    main()
