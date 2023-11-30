import pandas as pd
import matplotlib.pyplot as plt
import os



# Read the CSV file
file_paths = ['results/seg/results.csv', 'results/det/results.csv']  # Replace 'your_file.csv' with your CSV file path
data_seg = pd.read_csv(file_paths[0])
data_det = pd.read_csv(file_paths[1])

# Get the column names from the first row
column_names = list(data_seg.columns)


# Choose columns to plot (you can modify this based on your preferences)
columns_to_plot = [11, 12]  # Replace with indices of columns you want to plot

def plot_coloums(data: pd.DataFrame,
                 columns_to_plot: list,
                 column_names: list,
                 title: str = 'title' ,
                 ax_labels:tuple[str, str]=('x','y' ),
                 path="./out/temp.txt") -> None:
    # Get the column names from the first row
    column_names = list(data.columns)

    # Display column names to choose from
    print("Column names in the CSV file:")
    for idx, col_name in enumerate(column_names):
        print(f"{idx + 1}. {col_name}")
    plt.figure()
    # Plot selected columns
    for col_idx in columns_to_plot:
        plt.plot(data.iloc[:, col_idx], label=column_names[col_idx])

    # Add labels and title
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.title(title)

    # Show legend
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(path), f'{title}.png'), dpi=300)
    # Show the plot

# Plot selected columns

plot_coloums(data_seg, [11, 12], column_names, title='Segmentation mAPs', ax_labels=('Epochs','mAP' ), path=file_paths[0])
plot_coloums(data_seg, [1,2, 13, 14], column_names, title='Segmentation Losses', ax_labels=('Epochs','Loss' ), path=file_paths[0])
plot_coloums(data_det, [6,7], column_names, title='Detection mAPs', ax_labels=('Epochs','mAP' ), path=file_paths[1])
plot_coloums(data_det, [1, 8], column_names, title='Detection Losses', ax_labels=('Epochs','Loss' ), path=file_paths[1])
plt.show()
