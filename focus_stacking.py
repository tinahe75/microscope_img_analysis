import argparse
import cv2
import numpy as np
import glob
import os
from PIL import Image
from image_stitching import compute_translation, extract_tiff


def stack_images(image_files, img_width_downsample=1280, img_height_downsample=720, debug=False):
    """
    Focus stacking:
    First align images by calculating translation.
    Then, calculate the Laplacian for each image. High Laplacian = image region is in focus.
    For each pixel position in the output, read from input images that have high Laplacian values in that pixel position.

    :param image_files: SORTED list of image files
    :param img_width_downsample: calculate image translation & image focus using downsampled width
    :param img_height_downsample: calculate image translation & image focus using downsampled height
    :param debug: if TRUE, show translated images
    :return: stacked image
    """
    images = []
    focus_maps = []
    full_res_images = []
    orig_height, orig_width = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE).shape
    h_ratio = orig_height / img_height_downsample
    w_ratio = orig_width / img_width_downsample

    for f in image_files:
        img_orig = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        full_res_images.append(img_orig)
        img_rs = cv2.resize(img_orig, (img_width_downsample,img_height_downsample))
        images.append(img_rs)

    cumulative_translation_x = 0
    cumulative_translation_y = 0
    shifted_images = [images[0]]

    if debug:
        # show images after translation correction
        cv2.imshow('viz', shifted_images[0])
        cv2.waitKey(0)


    # use these to figure out final stacked image shape
    min_x, max_x = 0, orig_width
    min_y, max_y = 0, orig_height

    for i in range(len(images)-1):
        # calculate image translation using lower resolution images
        print(f"\nComputing translation between images {i + 1} and {i + 2}, using downsampled resolution w{img_width_downsample} x h{img_height_downsample}")
        translation_x, translation_y, matrix = compute_translation(images[i+1], images[i])
        cumulative_translation_x += translation_x
        cumulative_translation_y += translation_y
        matrix[0, 2] = cumulative_translation_x
        matrix[1, 2] = cumulative_translation_y
        rows, cols = images[i + 1].shape
        shifted_img = cv2.warpAffine(images[i+1], matrix[:2], (cols, rows))
        shifted_images.append(shifted_img)
        if debug:
            cv2.imshow('viz', shifted_img)
            cv2.waitKey(0)

        # apply translation to FULL RESOLUTION images
        matrix[0, 2] *= w_ratio
        matrix[1, 2] *= h_ratio
        rows, cols = full_res_images[i + 1].shape
        full_res_images[i+1] = cv2.warpAffine(full_res_images[i+1], matrix[:2], (cols, rows))

        # update boundaries
        max_x = min(max_x, int(orig_width + w_ratio * cumulative_translation_x))
        max_y = min(max_y, int(orig_height + h_ratio * cumulative_translation_y))
        min_x = max(min_x, int(w_ratio * cumulative_translation_x))
        min_y = max(min_y, int(h_ratio * cumulative_translation_y))

    # compute focus map using Gaussian blur -> Laplacian
    # Use lower resolution images! Then upsample the output focus map
    for ind, img_orig in enumerate(shifted_images):
        # you can adjust the Gaussian kernel size and sigma. Larger values -> smoother, less noise, but also blurry
        img = cv2.GaussianBlur(img_orig, (3,3), 1)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        print(f"\nVariance of Laplacian for image {ind}: {laplacian.var()}")
        fm = np.absolute(laplacian)
        fm = cv2.resize(fm, (orig_width, orig_height))
        focus_maps.append(fm)

    print("\n...... STACKING IMAGES ......")
    all_fm = np.stack(focus_maps, axis=-1)  # Shape: (h, w)
    print("\nstacked focus maps:", all_fm.shape)


    full_res_images_array = np.stack(full_res_images, axis=-1)  # Shape: (h, w, num_images)

    # For each pixel position, get the indices of the top 5 images.
    # For each pixel, we want the top images with highest Laplacian (sharpest focus)
    top_indices = np.argsort(all_fm, axis=-1)[..., -5:]  # Shape: (h, w, 5)

    # Use weights, higher weights for the highest Laplacian img, helps a bit with blending
    # You can adjust the weights, as well as how many images to use.
    # e.g. take only top 3 images, etc. Make sure weights sum to 1 though.
    weights = np.array([0.05, 0.1, 0.15, 0.2, 0.5], dtype=np.float32)

    # Gather the top 5 images per pixel for stacking
    top_images = np.take_along_axis(full_res_images_array, top_indices, axis=-1)  # Shape: (h, w, 5)

    # Apply weights and combine
    combined = np.sum(top_images * weights, axis=-1)  # Shape: (h, w)

    # Clip the values to [0, 255] and cast to uint8
    stacked_image = np.clip(combined, 0, 255).astype(np.uint8)

    # account for translation over time, only keep central region
    stacked_image = stacked_image[min_y:max_y, min_x:max_x]
    return stacked_image



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input folder or input TIFF file")
    parser.add_argument("--output_dir", type=str, help="folder to save output", default= ".")


    args = parser.parse_args()
    if os.path.isdir(args.input):
        all_imgs = sorted(glob.glob(os.path.join(args.input, "*.jpg")))
        folder_name = os.path.basename(args.input)
    elif args.input.lower().endswith(".tiff"):
        tmp_folder = extract_tiff(args.input)
        all_imgs = sorted(glob.glob(os.path.join(tmp_folder, "*.jpg")))
        folder_name = os.path.basename(tmp_folder)
    else:
        raise Exception("Invalid input! Provide a valid input folder or a tiff file.")

    stacked_img = stack_images(all_imgs)
    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{folder_name}_stacked.jpg")
    cv2.imwrite(out_file, stacked_img)
    print(f"output image saved to: {out_file}")