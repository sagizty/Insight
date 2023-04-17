"""
Inference Script Version Apr 17th 2023


"""

from dataclasses import dataclass

from tools import *
from unet import *
from DDPM import *


@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "Cifar-10"  # "MNIST", "Cifar-10", "Cifar-100", "Flowers"

    # Path to log inference images and save checkpoints
    root = "./Logs_Checkpoints"
    os.makedirs(root, exist_ok=True)

    # Current log and checkpoint directory.
    # by default start from "version_0", in training, given a value to a new name folder
    log_folder = None  # in inference: specific a folder name to load, by default will be the latest version
    checkpoint_name = "ddpm.tar"


@dataclass
class TrainingConfig:
    TIMESTEPS = 1000  # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32)
    NUM_EPOCHS = 2500
    BATCH_SIZE = 128
    LR = 2e-4

    NUM_WORKERS = 2 if str(BaseConfig.DEVICE) != "cpu" else 0  # 0 on cpu device


@dataclass
class ModelConfig:  # setting up attention unet
    BASE_CH = 64  # 64, 128, 256, 512
    BASE_CH_MULT = (1, 2, 4, 8)  # 32, 16, 8, 4
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 2  # 128


sd = Diffusion_setting(num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
                       img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE)

generate_video = True

# test
log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig(), inference=True)

model = UNet(
    input_channels=TrainingConfig.IMG_SHAPE[0],
    output_channels=TrainingConfig.IMG_SHAPE[0],
    base_channels=ModelConfig.BASE_CH,
    base_channels_multiples=ModelConfig.BASE_CH_MULT,
    apply_attention=ModelConfig.APPLY_ATTENTION,
    dropout_rate=ModelConfig.DROPOUT_RATE,
    time_multiple=ModelConfig.TIME_EMB_MULT,
)
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, BaseConfig.checkpoint_name), map_location='cpu')["model"], False)
model.to(BaseConfig.DEVICE)

inference(model, sd, img_shape=TrainingConfig.IMG_SHAPE, num_images=64, timesteps=1000, nrow=8,
          log_dir=log_dir, generate_video=generate_video, device=BaseConfig.DEVICE)
