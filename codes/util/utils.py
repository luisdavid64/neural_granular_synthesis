from PIL import Image
import os

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