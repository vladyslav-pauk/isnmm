# Install necessary libraries
# Uncomment below lines to install required packages if not already installed
# !pip install datasets --quiet
# !pip install git+https://github.com/MultimodalUniverse/MultimodalUniverse.git --quiet

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, load_dataset_builder
from scipy.ndimage import gaussian_filter1d

import mmu  # Ensure mmu is installed and imported correctly

# Initialize matplotlib
# %pylab inline

# **Image Samples**

# Load the Legacy Survey dataset
try:
    dset_ls = load_dataset("MultimodalUniverse/legacysurvey", streaming=True, split='train')
    dset_ls = dset_ls.with_format("numpy")
    dset_iterator = iter(dset_ls)

    # Fetch and inspect an example
    example = next(dset_iterator)
    print("Legacy Survey example keys:", example.keys())

    # Plotting image samples
    plt.figure(figsize=(12, 5))
    for i, b in enumerate(example['image']['band']):
        plt.subplot(1, 4, i + 1)
        plt.title(f'{b}')
        plt.imshow(example['image']['flux'][i], cmap='gray_r')
        plt.axis('off')
    plt.suptitle("Legacy Survey Image Samples")
    plt.show()
except Exception as e:
    print("Error loading or visualizing Legacy Survey dataset:", e)

# **Spectra Samples**

# Load the SDSS dataset
try:
    dset_sdss = load_dataset("MultimodalUniverse/sdss", streaming=True, split='train')
    dset_sdss = dset_sdss.with_format("numpy")
    dset_iterator = iter(dset_sdss)

    # Fetch and inspect an example
    example = next(dset_iterator)
    print("SDSS example keys:", example.keys())

    # Plotting spectra
    mask = example['spectrum']['lambda'] > 0
    plt.plot(example['spectrum']['lambda'][mask], example['spectrum']['flux'][mask])
    plt.title("Spectra Sample (SDSS)")
    plt.xlabel("Wavelength (lambda)")
    plt.ylabel("Flux")
    plt.show()
except Exception as e:
    print("Error loading or visualizing SDSS dataset:", e)

# **Time-Series Sample**

# Load the PLAsTiCC dataset
try:
    dset_plasticc = load_dataset("MultimodalUniverse/plasticc", streaming=True, split='train')
    dset_plasticc = dset_plasticc.with_format("numpy")
    dset_iterator = iter(dset_plasticc)

    # Fetch and inspect an example
    example = next(dset_iterator)
    print("PLAsTiCC example keys:", example.keys())

    # Plotting light curves
    plt.figure(figsize=(8, 5))
    for b in np.unique(example['lightcurve']['band']):
        mask = (example['lightcurve']['flux'] > 0) & (example['lightcurve']['band'] == b)
        plt.plot(example['lightcurve']['time'][mask], example['lightcurve']['flux'][mask], '+', label=b)
    plt.title(f"Light Curves (Object Type: {example['obj_type']})")
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend()
    plt.show()
except Exception as e:
    print("Error loading or visualizing PLAsTiCC dataset:", e)

# **Cross-Matching Across Modalities**

# Ensure local copies of datasets are available before proceeding
try:
    # Example dataset paths
    sdss_path = "data/MultimodalUniverse/v1/sdss"
    hsc_path = "data/MultimodalUniverse/v1/hsc"

    sdss = load_dataset_builder(sdss_path, trust_remote_code=True)
    hsc = load_dataset_builder(hsc_path, trust_remote_code=True)

    # Cross-match datasets
    dset = mmu.cross_match_datasets(sdss, hsc, matching_radius=1.0)
    print(f"Initial number of matches: {len(dset)}")

    # Format the dataset for processing
    dset = dset.with_format('numpy')

    # Example visualization
    example = dset[3]
    plt.figure(figsize=[15, 3])

    # Spectrum Plot
    plt.subplot(1, 6, 1)
    plt.ylim(0, 20)
    mask = example['spectrum']['lambda'] > 0
    plt.plot(example['spectrum']['lambda'][mask], example['spectrum']['flux'][mask], color='gray')
    plt.plot(example['spectrum']['lambda'][mask], gaussian_filter1d(example['spectrum']['flux'], sigma=5)[mask], color='k')
    plt.title('SDSS Spectrum')

    # Image Plots
    for i in range(5):
        plt.subplot(1, 6, i + 2)
        plt.imshow(np.log10(example['image']['flux'][i] + 2.), cmap='gray')
        plt.title(example['image']['band'][i])
        plt.axis('off')

    plt.suptitle("Cross-Matched Data Visualization")
    plt.show()
except Exception as e:
    print("Error during cross-matching or visualization:", e)