import imgaug.augmenters as iaa
import imghdr
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# ======================
# ✅ IMAGE DISPLAY FUNCTION
# ======================
def show_image(img, title="Result"):
    """Display image using matplotlib"""
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title(title)
    plt.show()


# ======================
# ✅ AUGMENTATION FUNCTIONS (NEW)
# ======================
def rotate_image(image, angle):
    """Safely rotate the image without losing face parts."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new bounding dimensions of the image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)

    return rotated


def shift_image(image, shift_x, shift_y):
    """
    Shift an image horizontally and vertically.

    Args:
        image: Input image (BGR)
        shift_x: Horizontal shift in pixels (positive = right, negative = left)
        shift_y: Vertical shift in pixels (positive = down, negative = up)

    Returns:
        Shifted image
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Create translation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Apply translation with border reflection to avoid black areas
    shifted = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT)

    return shifted





def adjust_brightness(image, factor=1.0):
    """
    Adjust the brightness of an image.

    Args:
        image: Input image (BGR).
        factor: Brightness factor.
                >1.0 for brighter, <1.0 for darker.

    Returns:
        Brightness adjusted image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # convert to HSV
    h, s, v = cv2.split(hsv)

    # Scale the V (brightness) channel
    v = np.clip(v * factor, 0, 255).astype(np.uint8)

    hsv_adjusted = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    return bright_img


def zoom_in_image(image, zoom_factor=1.2):
    """
    Zoom in on an image by the specified factor, cropping from the center.

    Args:
        image: Input image (BGR image).
        zoom_factor: Factor greater than 1.0 to zoom in. 
                     Example: 1.2 zooms in 20%.

    Returns:
        Zoomed-in image cropped to the original resolution.
    """
    height, width = image.shape[:2]

    # Calculate the scaled dimensions
    new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)

    # Resize the image
    scaled_image = cv2.resize(image, (new_width, new_height))

    # Calculate the center crop
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    cropped_image = scaled_image[start_y:start_y + height, start_x:start_x + width]

    return cropped_image


def safe_augment_face(image_path, output_dir, person_name, angles=(-10, -5, 0, 5, 10), zoom_factors=(1.1, 1.2)):
    """
    Augments an image by rotating, zooming in, and adjusting brightness and shifts, 
    and saves the results.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Unable to read image: {image_path} — Skipping...")
        return

    # Prepare the output folder for the person
    person_folder = os.path.join(output_dir, person_name)
    os.makedirs(person_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Get image dimensions for calculating shift amounts
    height, width = image.shape[:2]

    # Define shift parameters (as percentages of image dimensions)
    shift_percentages = [-0.1, 0, 0.1]  # -10%, 0%, and 10% shifts

    for angle in angles:
        try:
            rotated = rotate_image(image, angle)

            # Optionally resize the rotated image to match original dimensions
            rotated = cv2.resize(rotated, (image.shape[1], image.shape[0]))

            # Define output file path
            output_file = os.path.join(person_folder, f"{base_name}_angle{angle}.jpg")

            if os.path.exists(output_file):
                print(f"⚠️ File already exists, skipping: {output_file}")
                continue

            # Save the augmented image
            success = cv2.imwrite(output_file, rotated)

            if success:
                print(f"✅ Saved augmented image: {output_file}")
            else:
                print(f"❌ Failed to save image: {output_file}")

            # Add brightness variation
            brightness_factors = [0.8, 1.0, 1.2]  # Example factors
            for factor in brightness_factors:
                bright_image = adjust_brightness(rotated, factor)
                output_file = os.path.join(person_folder, f"{base_name}_angle{angle}_bright{factor}.jpg")
                cv2.imwrite(output_file, bright_image)

            # Add horizontal and vertical shifts
            for shift_x_pct in shift_percentages:
                for shift_y_pct in shift_percentages:
                    # Skip if both shifts are 0 (already handled by base image)
                    if shift_x_pct == 0 and shift_y_pct == 0:
                        continue

                    # Calculate pixel shifts based on image dimensions
                    shift_x = int(width * shift_x_pct)
                    shift_y = int(height * shift_y_pct)

                    # Apply the shift transformation
                    shifted_image = shift_image(rotated, shift_x, shift_y)

                    # Create a descriptive filename
                    x_direction = "right" if shift_x > 0 else "left" if shift_x < 0 else "center"
                    y_direction = "down" if shift_y > 0 else "up" if shift_y < 0 else "center"

                    # Define output file path for shifted image
                    shift_output_file = os.path.join(
                        person_folder,
                        f"{base_name}_angle{angle}_shift_{x_direction}{abs(shift_x)}_{y_direction}{abs(shift_y)}.jpg"
                    )

                    # Save the shifted image
                    success = cv2.imwrite(shift_output_file, shifted_image)

                    if success:
                        print(f"✅ Saved shifted image: {shift_output_file}")
                    else:
                        print(f"❌ Failed to save shifted image: {shift_output_file}")

                    # Optionally add brightness variations to shifted images too
                    for factor in brightness_factors:
                        if factor != 1.0:  # Skip default brightness
                            bright_shifted = adjust_brightness(shifted_image, factor)
                            bright_shift_output = os.path.join(
                                person_folder,
                                f"{base_name}_angle{angle}_shift_{x_direction}{abs(shift_x)}_{y_direction}{abs(shift_y)}_bright{factor}.jpg"
                            )
                            cv2.imwrite(bright_shift_output, bright_shifted)

        except Exception as e:
            print(f"❌ Error augmenting {image_path} with angle {angle}: {e}")

    # Add zoom-in augmentations
    for zoom_factor in zoom_factors:
        try:
            zoomed_image = zoom_in_image(image, zoom_factor)

            # Save the zoomed image
            zoom_output_file = os.path.join(person_folder, f"{base_name}_zoom{zoom_factor}.jpg")
            success = cv2.imwrite(zoom_output_file, zoomed_image)
            
            if success:
                print(f"✅ Saved zoomed image: {zoom_output_file}")
            else:
                print(f"❌ Failed to save zoomed image: {zoom_output_file}")
            
            # Optionally, add brightness variations to the zoomed images
            for factor in brightness_factors:
                bright_zoomed = adjust_brightness(zoomed_image, factor)
                bright_zoom_output = os.path.join(person_folder, f"{base_name}_zoom{zoom_factor}_bright{factor}.jpg")
                cv2.imwrite(bright_zoom_output, bright_zoomed)

        except Exception as e:
            print(f"❌ Error zooming {image_path} with factor {zoom_factor}: {e}")





def create_augmented_dataset(input_dir, output_dir):
    """
    Create an augmented dataset by applying rotations to images.
    Reads from input_dir and writes augmented images to output_dir.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory does not exist: {input_dir}")
        return

    # Traverse each person's folder in the input directory
    for person_name in os.listdir(input_dir):
        person_folder_path = os.path.join(input_dir, person_name)

        # Skip if not a directory
        if not os.path.isdir(person_folder_path):
            print(f"⚠️ Skipping non-directory item: {person_folder_path}")
            continue

        print(f"📂 Processing folder for person: {person_name}")

        # Traverse each image file in the person's folder
        for filename in os.listdir(person_folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_folder_path, filename)
                print(f"ℹ️ Found image: {image_path}")

                try:
                    # Perform augmentation
                    safe_augment_face(image_path, output_dir, person_name)
                except Exception as e:
                    print(f"❌ Error processing image {image_path}: {e}")
            else:
                print(f"⚠️ Skipping unsupported file: {filename}")


# Alternative function using imgaug library for more advanced augmentations
def augment_with_imgaug(image_path, output_dir, person_name):
    """
    Augment an image using the imgaug library for more sophisticated transformations.
    This demonstrates an alternative approach to the manual transformations.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Unable to read image: {image_path} — Skipping...")
        return

    # Prepare the output folder for the person
    person_folder = os.path.join(output_dir, person_name)
    os.makedirs(person_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Define augmentation sequence with height and width shifts
    augmentation = iaa.Sequential([
        iaa.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},  # Shift by -15% to 15%
            rotate=(-15, 15),  # Rotate by -15 to 15 degrees
            mode='reflect'  # Use reflection padding to avoid black areas
        ),
        iaa.Sometimes(0.5, iaa.MultiplyBrightness((0.8, 1.2)))  # 50% chance to adjust brightness
    ])

    # Generate multiple augmented versions
    for i in range(10):  # Create 10 variations
        try:
            # Apply augmentation
            augmented_image = augmentation(image=image)

            # Save the augmented image
            output_file = os.path.join(person_folder, f"{base_name}_imgaug_{i}.jpg")
            success = cv2.imwrite(output_file, augmented_image)

            if success:
                print(f"✅ Saved imgaug augmented image: {output_file}")
            else:
                print(f"❌ Failed to save imgaug augmented image: {output_file}")

        except Exception as e:
            print(f"❌ Error with imgaug augmentation {i} for {image_path}: {e}")


# if __name__ == "__main__":
#     input_dir = "C:/Users/zeyad/PyCharmMiscProject/known_faces"  # Replace this with the actual path
#     output_dir = "C:/Users/zeyad/PyCharmMiscProject/augmented_faces"  # Replace this with the actual path
#
#     # Run the dataset augmentation
#     create_augmented_dataset(input_dir, output_dir)