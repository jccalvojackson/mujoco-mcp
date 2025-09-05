import numpy as np
from PIL import Image as PILImage


def get_images_as_grid(
    images: list[np.ndarray],
) -> PILImage:
    """Concatenate multiple images in a grid layout with WebP compression."""
    import math

    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images provided")

    MAX_IMAGES = 9
    if num_images > MAX_IMAGES:
        raise ValueError(f"Too many images: {num_images} > {MAX_IMAGES}")

    # Calculate grid dimensions for a more square layout
    if num_images <= 4:
        cols = min(2, num_images)
        rows = math.ceil(num_images / cols)
    else:
        # For 5+ images, prefer 3 columns for better aspect ratio
        cols = 3
        rows = math.ceil(num_images / cols)

    # Get dimensions of individual images
    img_height, img_width = images[0].shape[:2]

    # Create the grid canvas
    grid_width = cols * img_width
    grid_height = rows * img_height
    concat_image: PILImage = PILImage.new("RGB", (grid_width, grid_height))

    # Place images in grid
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        x_pos = col * img_width
        y_pos = row * img_height
        concat_image.paste(PILImage.fromarray(image), (x_pos, y_pos))

    return concat_image
