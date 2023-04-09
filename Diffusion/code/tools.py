"""
Tools Script Version Apr 10th 2023


"""

import os
import cv2
import base64
from PIL import Image

import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as TF
import torchvision.datasets as datasets
from IPython.display import display, HTML


def setup_log_directory(config, inference=False):
    """
    Log and Model checkpoint directory Setup

    :param config: BaseConfig
    :param inference: True if this is used in the inference code
    :return:
    """

    if config.log_folder is not None:
        version_name = config.log_folder
    else:
        status = False
        # already has previous version
        if os.path.isdir(config.root):
            # Get all folders numbers in the root_log_dir
            folder_numbers =[]
            for file in os.listdir(config.root):
                try:
                    version_idx = int(file.replace("version_", ""))
                except:
                    pass
                else:
                    folder_numbers.append(version_idx)

            if len(folder_numbers) == 0:
                status = True
            else:
                # Find the latest version number present in the log_dir
                last_version_number = max(folder_numbers)
                # New version name
                version_name = f"version_{last_version_number + 1}" if not inference \
                    else f"version_{last_version_number}"
        else:
            status = True

        # no previous version
        if status:
            if inference:
                print('no trained model')
                return None
            else:
                os.makedirs(config.root, exist_ok=True)
                version_name = 'version_0'

    # Update the training config default directory
    log_dir = os.path.join(config.root, version_name, "Training_inference") if not inference \
        else os.path.join(config.root, version_name, "Inference_results")
    checkpoint_dir = os.path.join(config.root, version_name, "checkpoints")

    # Create new directory for saving new experiment version, if already there (in inference) auto skip
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")

    return log_dir, checkpoint_dir


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get(element: torch.Tensor, idxs: torch.Tensor):
    """
    Get values from "element" by index positions (idxs) and
    reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, idxs)  # size: B (same as idxs)
    return ele.reshape(-1, 1, 1, 1)  # size: B,1,1,1


def frames2vid(images, save_path):
    WIDTH = images[0].shape[1]
    HEIGHT = images[0].shape[0]

    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     fourcc = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))

    # Appending the images to the video one by one
    for image in images:
        video.write(image)

    # Deallocating memories taken for window creation
    #     cv2.destroyAllWindows()
    video.release()
    return


def display_gif(gif_path):
    b64 = base64.b64encode(open(gif_path, 'rb').read()).decode('ascii')
    display(HTML(f'<img src="data:image/gif;base64,{b64}" />'))


def get_dataset(dataset_name='MNIST'):
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((32, 32),
                      interpolation=TF.InterpolationMode.BICUBIC,
                      antialias=True),
            #             TF.RandomHorizontalFlip(),
            TF.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
        ]
    )

    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-10":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        dataset = datasets.ImageFolder(root="/kaggle/input/flowers-recognition/flowers", transform=transforms)

    return dataset


def get_dataloader(dataset_name='MNIST', batch_size=32, pin_memory=False, shuffle=True, num_workers=0, device="cpu"):
    dataset = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                            shuffle=shuffle)
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader


def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0
