from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

def stitch_images_dir(img_dir="umap_exp", output_filename='umap_feature_grid.png'):
    # List all PNG files in the directory
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    img_files.sort()  # Optional: sort alphabetically

    # Load images
    images = [Image.open(os.path.join(img_dir, f)) for f in img_files]

    # Determine grid size (e.g., 2 columns)
    n_cols = 4
    n_rows = (len(images) + n_cols - 1) // n_cols

    # Get image size (assume all images are the same size)
    img_w, img_h = images[0].size

    # Create a blank canvas
    stitched_img = Image.new('RGB', (n_cols * img_w, n_rows * img_h), (255, 255, 255))

    # Paste images into the canvas
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        stitched_img.paste(img, (col * img_w, row * img_h))

    # Save or show the result
    stitched_img.save(output_filename)
    stitched_img.show()

def plot_umap(all_grains_2d, grain_labels, output_path, feature_list, show=True):
    unique_labels = sorted(set(grain_labels))
    palette = sns.color_palette("tab10", len(unique_labels))
    label_to_color = {lab: palette[i] for i, lab in enumerate(unique_labels)}

    # Plot UMAP with colors
    plt.figure(figsize=(10, 10))
    for lab in unique_labels:
        idx = [i for i, l in enumerate(grain_labels) if l == lab]
        plt.scatter(all_grains_2d[idx, 0], all_grains_2d[idx, 1], s=20, alpha=0.6, label=lab, color=label_to_color[lab])
    # add features to title
    plt.title(f"UMAP of Audio Grains (Labeled) - Features: {', '.join(feature_list)}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.axis('equal')
    if show:
        plt.show()
    # Save fig name based on features used
    plt.savefig(os.path.join(output_path, f"umap_{'_'.join(feature_list)}.png"))
    plt.close()