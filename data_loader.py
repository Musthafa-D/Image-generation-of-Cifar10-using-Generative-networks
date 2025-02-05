from ccbdl.data.utils.get_loader import get_loader
from torchvision import transforms
import random
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The default value of the antialias parameter of all the resizing transforms")


def prepare_data(data_config):
    # augmentations_list = get_augmentation(data_config['augmentation'])
    # print(augmentations_list)
    # final_transforms = transforms.Compose(augmentations_list)

    # data_config["transform_input"] = final_transforms

    loader = get_loader(data_config["dataset"])
    train_data, test_data, val_data = loader(
        **data_config).get_dataloader()

    
    if sys.platform == "win32":
        view_data(train_data, data_config)
        view_data(test_data, data_config)

    return train_data, test_data, val_data

def get_augmentation(augmentations):
    transform_list = []
    for item in augmentations:
        if isinstance(item, str):  # Direct transform like RandomHorizontalFlip
            transform = getattr(transforms, item)()
            transform_list.append(transform)
        elif isinstance(item, dict):  # Transform with parameters like RandomRotation
            for name, params in item.items():
                if isinstance(params, list):  # If parameters are given as a list
                    transform = getattr(transforms, name)(*params)
                else:  # If a single parameter is given
                    transform = getattr(transforms, name)(params)
                transform_list.append(transform)
    return transform_list

def view_data(data, data_config):
    # View the first image in train_data or test_data
    batch = next(iter(data))
    inputs, labels = batch

    # Set up the subplot dimensions
    fig, axs = plt.subplots(2, 5, figsize=(15, 7))
    axs = axs.ravel()

    for i in range(10):
        idx = random.randint(0, data_config["batch_size"]-1)
        image = inputs[idx]
        label = labels[idx].item()  # Convert the label tensor to an integer

        image_np = image.permute(1, 2, 0).numpy()

        class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                      6: "frog", 7: "horse", 8: "ship", 9: "truck"}

        # Display the image along with its label in the subplot
        axs[i].imshow(image_np)
        axs[i].set_title(f"{class_dict[label]}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
