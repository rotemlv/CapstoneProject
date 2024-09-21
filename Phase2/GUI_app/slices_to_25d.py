import cv2
import ml_collections
import numpy as np
import os

import torch
import torch_directml

dml = torch_directml.device(torch_directml.default_device())
from my_model import VisionTransformer

DATASET_PATH = 'D:/kaggle/input/uw-madison-gi-tract-image-segmentation/train/case30/case30_day1/scans/'


def convert_slices_to_npy(slice_paths, output_shape=(224, 224)):
    # Read and resize each slice
    slices = []
    for slice_path in slice_paths:
        slice_path = DATASET_PATH + slice_path
        img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, output_shape)
        slices.append(resized_img)

    # Stack the slices along the channel dimension
    stacked_slices = np.stack(slices, axis=-1)
    # Normalize the pixel values to [0, 1]
    normalized_slices = stacked_slices.astype(np.float32) / 255.0

    return normalized_slices

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = "https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/R50%2BViT-B_16.npz"
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config

def segment_with_model(input_array):
    # Load the model
    device = dml#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_vit = get_r50_b16_config()
    config_vit.n_classes = 3
    config_vit.n_skip = 3
    config_vit.patches.grid = (
        # this is supposed to be ImageSizeW / vit-patch-size, ImageSizeH / vit-patch-size, but I can't
        int(224 / 16), int(224 / 16))
    model = VisionTransformer(config_vit, img_size=224, num_classes=3)
    model.load_state_dict(torch.load("my_weights.bin", map_location=torch.device("cpu")))

    model.to(device)
    model.eval()

    # Prepare input tensor
    input_tensor = torch.from_numpy(np.transpose(input_array, (2, 0, 1))).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Convert output to numpy array
    segmentation_mask = torch.argmax(outputs, dim=1).cpu().numpy()[0]

    return segmentation_mask


# Example usage
slice_paths = [
    'slice_0069_266_266_1.50_1.50.png',
    'slice_0070_266_266_1.50_1.50.png',
    'slice_0071_266_266_1.50_1.50.png'
]

output_array = convert_slices_to_npy(slice_paths)

print(f"Output shape: {output_array.shape}")
print(f"Dtype: {output_array.dtype}")

# Save the array as an npy file
np.save('output_slices.npy', output_array)

# Load the saved npy file to verify
loaded_array = np.load('output_slices.npy')
print(f"Loaded array shape: {loaded_array.shape}")
print(f"Loaded array dtype: {loaded_array.dtype}")

# Convert slices to numpy array
input_array = convert_slices_to_npy(slice_paths)

print(f"Input shape: {input_array.shape}")
print(f"Dtype: {input_array.dtype}")
# Segment using the model
segmentation_mask = segment_with_model(input_array)

print(f"Segmentation mask shape: {segmentation_mask.shape}")
print(f"Segmentation mask dtype: {segmentation_mask.dtype}")

# Visualize the segmentation result
import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.imshow(input_array[:, :, 0], cmap='gray')
# plt.title("Input Slice")
#
# plt.subplot(1, 2, 2)
# plt.imshow(segmentation_mask, cmap='viridis')
# plt.title("Segmentation Mask")
#
# plt.tight_layout()
# plt.show()
#
# # Save the segmentation mask as an npy file
# np.save('segmentation_result.npy', segmentation_mask)
#
# print("Segmentation result saved as segmentation_result.npy")



def plot_segmentation_results(input_array, segmentation_mask):
    """
    Plot the input image alongside the segmentation mask.

    Args:
    input_array (numpy array): Input image data
    segmentation_mask (numpy array): Segmentation mask

    Returns:
    None
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    # Plot the input image
    axes[0].imshow(input_array[:, :, 0], cmap='gray')
    axes[0].set_title("Input Slice")
    axes[0].axis('off')  # Turn off axis ticks

    # Plot the segmentation mask on top of the input image
    axes[1].imshow(input_array[:, :, 0], cmap='viridis', vmin=0, vmax=2)

    # Add colorbar manually
    norm = plt.Normalize(vmin=0, vmax=2)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], orientation='vertical')

    # Set colorbar ticks and labels
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Background', 'Large Bowel', 'Small Bowel/Stomach'])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Usage example
if __name__ == "__main__":
    # Assuming you have already loaded your data
    slice_paths = [
        'slice_0069_266_266_1.50_1.50.png',
        'slice_0070_266_266_1.50_1.50.png',
        'slice_0071_266_266_1.50_1.50.png'
    ]

    # Convert slices to numpy array
    input_array = convert_slices_to_npy(slice_paths)

    # Segment using the model
    segmentation_mask = segment_with_model(input_array)

    # Plot the results
    plot_segmentation_results(input_array, segmentation_mask)
