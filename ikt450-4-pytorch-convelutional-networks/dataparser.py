from torch import Tensor, layout
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data


def load_images(path="./data_true/training/", resize_to=255, center_crop_to=255):
    transformer = torchvision.transforms.Compose(
                            transforms=[
                                        torchvision.transforms.Resize(resize_to),
                                        torchvision.transforms.CenterCrop(center_crop_to),
                                        torchvision.transforms.ToTensor(),
                                       ])

    training_set = torchvision.datasets.ImageFolder(path, transformer)

    return training_set

def get_sub_set(dataset: torchvision.datasets.folder.ImageFolder,
                procent: float) -> torchvision.datasets.folder.ImageFolder:
    """
    Returns a subset of the dataset based on the procent passed in.
    The datapoints are picked at random though out the set. 

    To get a subset, we can use the subset method. 
    How ever, to use this, we need a collection of indeicers for the elements
    we want to keep
    """
    
    length_of_subset = int(len(dataset)*procent)
    # Create a collection of randem indexes, the same length as dataset. 
    idx = torch.randperm(len(dataset))
    # Create subset, only picking the elements at index idx, untill length of subset 

    
    sub =  torch.utils.data.Subset(dataset=dataset, 
        indices=idx[0:length_of_subset].tolist())
    return sub
    


def main():
    dataset = get_sub_set(load_images(), 0.4)
    print(type(dataset))
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=32,
                                                  shuffle=True) 
    images, labels = next(iter(loader))
    show_images(images=images, labels=labels, cols=4, rows=8)

if __name__ == "__main__":
    main()
