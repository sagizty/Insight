"""
Training   Script Version Apr 17th 2023


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


# view dataset
'''
# get_dataloader
loader = get_dataloader(
    dataset_name=BaseConfig.DATASET,
    batch_size=128,
    device='cpu',
)

plt.figure(figsize=(12, 6), facecolor='white')
for b_image, _ in loader:
    b_image = inverse_transform(b_image).cpu()
    grid_img = make_grid(b_image / 255.0, nrow=16, padding=True, pad_value=1, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    break
plt.show()
'''

# diffusion process example
'''
sd = SimpleDiffusion(num_diffusion_timesteps=TrainingConfig.TIMESTEPS, device="cpu")

loader = iter(  # converting dataloader into an iterator for now.
    get_dataloader(
        dataset_name=BaseConfig.DATASET,
        batch_size=6,
        device="cpu",
    )
)
x0s, _ = next(loader)

noisy_images = []
specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

for timestep in specific_timesteps:
    timestep = torch.as_tensor(timestep, dtype=torch.long)

    xts, _ = forward_diffusion(sd, x0s, timestep)
    xts = inverse_transform(xts) / 255.0
    xts = make_grid(xts, nrow=1, padding=1)

    noisy_images.append(xts)

# Plot and see samples at different timesteps

_, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor='white')

for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
    ax[i].imshow(noisy_sample.squeeze(0).permute(1, 2, 0))
    ax[i].set_title(f"t={timestep}", fontsize=8)
    ax[i].axis("off")
    ax[i].grid(False)

plt.suptitle("Forward Diffusion Process", y=0.9)
plt.axis("off")
plt.show()
'''
model = UNet(
    input_channels=TrainingConfig.IMG_SHAPE[0],
    output_channels=TrainingConfig.IMG_SHAPE[0],
    base_channels=ModelConfig.BASE_CH,
    base_channels_multiples=ModelConfig.BASE_CH_MULT,
    apply_attention=ModelConfig.APPLY_ATTENTION,
    dropout_rate=ModelConfig.DROPOUT_RATE,
    time_multiple=ModelConfig.TIME_EMB_MULT,
)
model.to(BaseConfig.DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)

dataloader = get_dataloader(
    dataset_name=BaseConfig.DATASET,
    batch_size=TrainingConfig.BATCH_SIZE,
    device=BaseConfig.DEVICE,
    pin_memory=True,
    num_workers=TrainingConfig.NUM_WORKERS,
)

loss_fn = nn.MSELoss()

sd = Diffusion_setting(num_diffusion_timesteps=TrainingConfig.TIMESTEPS,
                       img_shape=TrainingConfig.IMG_SHAPE, device=BaseConfig.DEVICE)

scaler = amp.GradScaler()

log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())

generate_video = False

train(model, sd, dataloader, optimizer, scaler, loss_fn, img_shape=TrainingConfig.IMG_SHAPE,
      total_epochs=TrainingConfig.NUM_EPOCHS, timesteps=TrainingConfig.TIMESTEPS, log_dir=log_dir,
      checkpoint_dir=checkpoint_dir, generate_video=generate_video, device=BaseConfig.DEVICE,
      checkpoint_name=BaseConfig.checkpoint_name)
