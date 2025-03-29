from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch

# Load the dataset
dset = load_dataset('MultimodalUniverse/manga', split='train', streaming=False)

# Convert dataset to a NumPy-compatible format
dset_ls = dset.with_format("numpy")

# Initialize metadata lists
image_shapes = []
meta_info = []

# Print dimensions and gather metadata for the first few examples
for idx, example in enumerate(dset_ls):
    if 'images' in example:
        for img in example['images']:
            if 'flux' in img:
                # Record shape and other metadata
                image_shapes.append(np.array(img['flux']).shape)
                meta_info.append(img.get('meta', {}))  # Adjust 'meta' if necessary

                # Print metadata for the first few images
                if idx < 5:
                    print(f"Image Shape: {image_shapes[-1]}, Metadata: {meta_info[-1]}")

    # Stop after a few examples to avoid overloading
    if idx >= 5:
        break

# Print unique image shapes and metadata summary
print(f"\nUnique Image Shapes: {set(image_shapes)}")
print(f"Total Images Processed: {len(image_shapes)}")

# Gather images into a tensor hypercube
images_list = []
for example in dset_ls:
    if 'images' in example:
        image_batch = [img['flux'] for img in example['images'] if 'flux' in img]
        images_list.extend(image_batch)

# Convert list of images to a 4D tensor (Batch x Channels x Height x Width)
image_tensor = torch.tensor(np.array(images_list))  # Ensure shapes are uniform
print(f"\nTensor Shape: {image_tensor.shape}")  # e.g., (Batch, H, W) or (Batch, C, H, W)

# Plot a few images
plt.figure(figsize=(12, 5))
for i in range(min(len(image_tensor), 4)):  # Show up to 4 images
    plt.subplot(1, 4, i + 1)
    plt.title(f'Image {i + 1}')
    plt.imshow(image_tensor[i], cmap='gray_r')
    plt.axis('off')
plt.show()