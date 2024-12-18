import glob
import os
import cv2
import numpy as np
import argparse
from PIL import Image

def extract_tiff(tiff_path, skip_initial_pages=-1):
    print("extracting tiff file:", tiff_path)
    orig_name = os.path.splitext(os.path.basename(tiff_path))[0]
    os.makedirs(orig_name, exist_ok=True)
    with Image.open(tiff_path) as tiff_image:
        page_number = 0
        while True:
            # Save each page as a separate file
            output_path = os.path.join(orig_name, f"{orig_name}_page_{page_number:04d}.jpg")
            if page_number > skip_initial_pages:
                tiff_image.save(output_path, format="JPEG")
                print(f"Saved: {output_path}")

            page_number += 1

            # Go to the next frame/page
            try:
                tiff_image.seek(page_number)
            except EOFError:
                # No more pages
                break
    print(f"tiff extracted to folder: {orig_name}\n")
    return orig_name

def compute_translation(img1, img2):
    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate the affine transformation matrix
    matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Extract the translation components
    translation_x = matrix[0, 2]
    translation_y = matrix[1, 2]

    print(f"Translation: x = {translation_x}, y = {translation_y}")
    return translation_x, translation_y, matrix

def get_hist_highest_freq_bin(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    max_bin = np.argmax(hist)  # Bin corresponding to the maximum frequency
    # print(max_bin)
    return max_bin

def stitch_images(all_imgs, overlap_ratio, img_width_downsample=1600, img_height_downsample=900, adjust_brightness=True, debug=False):
    """
    Stitch together 9 images. Calculate the offset for each image, and place together on a canvas.
    Assumes that the camera movement was like this:

    7 -> 8 -> 9
    ↑
    6 <- 5 <- 4
              ↑
    1 -> 2 -> 3

    :param all_imgs: list of 9 image paths. Should be SORTED based on camera movement.
    :param overlap_ratio: e.g. 0.25 if images have 25% overlap.
    :param img_width_downsample: downsampled image width used for computing translation
    :param img_height_downsample: downsampled image height used for computing translation
    :param adjust_brightness: automatically adjust brightness for image pairs using img histograms
    :param debug: if TRUE, show aligned overlap region for image pairs
    :return: stitched image
    """

    if len(all_imgs) != 9:
        raise Exception("ERROR: Image list should contain 9 items!")

    orig_height, orig_width = cv2.imread(all_imgs[0], cv2.IMREAD_GRAYSCALE).shape
    h_ratio = orig_height / img_height_downsample
    w_ratio = orig_width / img_width_downsample

    w_overlap = int(img_width_downsample * overlap_ratio)
    h_overlap = int(img_height_downsample * overlap_ratio)

    full_res_images = []
    direction_map = ['r', 'r', 'u', 'l', 'l', 'u', 'r', 'r']  # camera movement direction

    # variables used for tracking image coordinates
    translation_x_cumulative = 0
    translation_y_cumulative = 0
    position_offset_x = 0
    position_offset_y = 0

    # (position_offset_x, position_offset_y) is the offset for the FIRST image (its top left corner).
    # The first image is located in the bottom left corner of the canvas.
    x1, y1 = position_offset_x, position_offset_y
    x2, y2 = position_offset_x + img_width_downsample, position_offset_y + img_height_downsample
    print(f"image 1 coordinates: x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")

    # use this list to record the offset for each image
    offsets = [(position_offset_x, position_offset_y)]

    # use these to figure out how big the canvas has to be
    min_x = x1
    min_y = y1
    max_x = x2
    max_y = y2

    for i in range(8):
        print("\n")
        img1_orig = cv2.imread(all_imgs[i], cv2.IMREAD_GRAYSCALE)
        img2_orig = cv2.imread(all_imgs[i + 1], cv2.IMREAD_GRAYSCALE)

        if len(full_res_images) == 0:
            full_res_images.append(img1_orig)
            base_bin = get_hist_highest_freq_bin(img1_orig)

        if adjust_brightness:
            # simple brightness adjustment: calculate histogram for each image
            # use image1 as reference. Get its histogram bin with max frequency.
            # scale the current image so that its max frequency bin aligns with image1's max frequency bin
            curr_bin = get_hist_highest_freq_bin(img2_orig)
            diff = base_bin / curr_bin
            img2_orig = np.clip(img2_orig.astype(float) * diff,0,255).astype(np.uint8)
            print(f"scaling current image pixel intensity by factor = {diff}")

        # resize images to smaller resolutions before computing translation
        img1_rs = cv2.resize(img1_orig, (img_width_downsample, img_height_downsample))
        img2_rs = cv2.resize(img2_orig, (img_width_downsample, img_height_downsample))

        if direction_map[i] == 'r':
            img1 = img1_rs[:, img_width_downsample - w_overlap:]
            img2 = img2_rs[:, :w_overlap]
            position_offset_x += img_width_downsample - w_overlap
        elif direction_map[i] == 'u':
            img1 = img1_rs[:h_overlap, :]
            img2 = img2_rs[img_height_downsample - h_overlap:, :]
            position_offset_y -= img_height_downsample - h_overlap
        else:
            img1 = img1_rs[:, :w_overlap]
            img2 = img2_rs[:, (img_width_downsample - w_overlap):]
            position_offset_x -= img_width_downsample - w_overlap

        full_res_images.append(img2_orig)


        print(f"Computing translation between images {i+1} and {i+2}, using downsampled resolution w{img_width_downsample} x h{img_height_downsample}")
        translation_x, translation_y, matrix = compute_translation(img1, img2)

        if debug:
            rows, cols = img1.shape
            shifted_img1 = cv2.warpAffine(img1, matrix[:2], (cols, rows))
            cv2.imshow('viz', shifted_img1)
            cv2.imshow('viz2', img2)
            cv2.waitKey(0)

        # track cumulative translation, relative to the FIRST image
        translation_x_cumulative += translation_x
        translation_y_cumulative += translation_y

        # calculate where this image will be placed on the canvas
        x1 = int(position_offset_x - translation_x_cumulative)
        y1 = int(position_offset_y -translation_y_cumulative)
        offsets.append((x1, y1))
        x2 = x1 + img_width_downsample
        y2 = y1 + img_height_downsample

        print(f"image {i+2} coordinates: x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}")

        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)


    canvas_width = int((max_x-min_x) * w_ratio)
    canvas_height = int((max_y-min_y) * h_ratio)
    print(f"\nOutput canvas size: w{canvas_width} x h{canvas_height}")
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # stitch all images together, put onto canvas
    for i in range(9):
        x_offset, y_offset = offsets[-i-1]
        x_offset -= min_x
        y_offset -= min_y
        ys = int(y_offset * h_ratio)
        ye = int((y_offset + img_height_downsample)* h_ratio)
        xs = int(x_offset * w_ratio)
        xe = int((x_offset + img_width_downsample) * w_ratio)
        canvas[ys:ye, xs:xe] = full_res_images[-1-i]

    view_width = 1600
    view_height = int(view_width / canvas_width * canvas_height)
    cv2.imshow('viz', cv2.resize(canvas, (view_width, view_height)))
    cv2.waitKey(0)
    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input folder or input TIFF file")
    parser.add_argument("--output_dir", type=str, help="folder to save output", default= ".")
    parser.add_argument("--overlap", type=float, required=True, help="overlap ratio, between 0 and 1. "
                                                                     "WARNING: Works best if the overlap ratio matches with your actual overlap ratio. "
                                                                     "You will get bad alignment when the values don't match."
                                                                     "Recommendations: use 0.25 ~ 0.3 for 30% overlap, use 0.35 ~ 0.4 for 40% overlap, etc.")

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

    canvas = stitch_images(all_imgs, args.overlap)
    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{folder_name}_merged.jpg")
    cv2.imwrite(out_file, canvas)
    print(f"output image saved to: {out_file}")
