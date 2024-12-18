import cv2
import glob
import os
import numpy as np
import argparse
import shutil

def get_gradient_map(image):
    image = cv2.GaussianBlur(image, (3,3), 3)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  

    # Compute gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    gradient = np.uint8(normalized)
    # cv2.imshow("Gradient", gradient)
    # cv2.waitKey(0)
    return gradient

def align_img_pair_ecc(im1, im2, warp_mode=cv2.MOTION_TRANSLATION, use_gradient=True):
    if len(im1.shape) == 3:
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        im1_gray = im1
        im2_gray = im2

    if use_gradient:
        im1_gray = get_gradient_map(im1_gray)
        im2_gray = get_gradient_map(im2_gray)

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000 

    # Specify the threshold of the increment 
    # in the correlation coefficient between two iterations
    termination_eps = 1e-2 #1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
        horz_shift = warp_matrix[0][-1]
        vert_shift = warp_matrix[1][-1]
        print(f"Horizontal shift: {horz_shift:.2f}, vertical shift: {vert_shift:.2f}")
    except:
        raise Exception("no correlation between images!")

    return warp_matrix

def align_sequence(all_imgs, autocrop=True, img_width_downsample=800, img_height_downsample=450):
    """
    Align a sequence of images using ECC algorithm
    :param all_imgs: list of SORTED image file paths
    :param autocrop: if TRUE, automatically crop images to remove black borders
    :param img_width_downsample: downsampled image width used for computing translation
    :param img_height_downsample: downsampled image height used for computing translation
    :return: None
    """
    # use smaller images to calculate alignment, much faster
    downsample_resolution = (img_width_downsample, img_height_downsample)  
    tmp_img = cv2.imread(all_imgs[0], 0)
    orig_resolution = (tmp_img.shape[1],tmp_img.shape[0])  # original resolution of input images
    scale_factor = orig_resolution[0] / downsample_resolution[0]
    total_horz_shift, total_vert_shift = 0, 0

    # keep track of max horizontal and vertical shift (up, down, left, right)
    max_shift_horz, max_shift_vert = 0,0
    max_shift_horz_neg, max_shift_vert_neg = 0, 0
    saved = []

    # all alignment is relative to the first image in the sequence
    for i in range(0, len(all_imgs) - 1):
        print(f"\n.... [{i+1} / {len(all_imgs) - 1}] Processing {all_imgs[i]} ....")
        im1_orig = cv2.imread(all_imgs[i], 0)
        if i == 0:
            shutil.copy(all_imgs[0], out_dir)
            saved.append(os.path.join(out_dir, os.path.basename(all_imgs[0])))
        im2_orig = cv2.imread(all_imgs[i + 1], 0)
        im1 = cv2.resize(im1_orig, downsample_resolution)
        im2 = cv2.resize(im2_orig, downsample_resolution)

        warp_matrix = align_img_pair_ecc(im1, im2)
        # calculate the cumulative shift, relative to first image
        total_horz_shift += warp_matrix[0, -1] * scale_factor  
        total_vert_shift += warp_matrix[1, -1] * scale_factor
        warp_matrix[0, -1] = total_horz_shift
        warp_matrix[1, -1] = total_vert_shift
        max_shift_horz = max(max_shift_horz, total_horz_shift)
        max_shift_horz_neg = min(max_shift_horz_neg, total_horz_shift)
        max_shift_vert = max(max_shift_vert, total_vert_shift)
        max_shift_vert_neg = min(max_shift_vert_neg, total_vert_shift)
        im2_aligned = cv2.warpAffine(im2_orig, warp_matrix, orig_resolution, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        out_file = os.path.join(out_dir, os.path.basename(all_imgs[i+1]))
        cv2.imwrite(out_file, im2_aligned)
        print(f"Saved output to: {out_file}")
        saved.append(out_file)

    print(f"\n\nmax h shift: {max_shift_horz_neg} ~ {max_shift_horz}")
    print(f"max v shift: {max_shift_vert_neg} ~ {max_shift_vert}")
    if autocrop:  
        # remove black border created by warpAffine
        xs = - int(np.ceil(max_shift_horz_neg))
        xe = orig_resolution[0] - int(np.ceil(max_shift_horz))
        ys = - int(np.ceil(max_shift_vert_neg))
        ye = orig_resolution[1] - int(np.ceil(max_shift_vert))
        print(f"Using the crop window X: {xs} ~ {xe}, Y: {ys} ~ {ye}\n")
        for im in saved:
            img = cv2.imread(im)
            cv2.imwrite(im, img[ys:ye, xs:xe, :])
            print(f"Saved corrected output to: {im}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Align a series of images")
    parser.add_argument("-i", "--input", type=str, required=True, help="input directory")
    parser.add_argument("-o", "--output", type=str, required=True, help="output directory")
    parser.add_argument("--autocrop", default=False, action="store_true", help="automatically crop images")
    args = parser.parse_args()

    img_folder = args.input
    out_dir = args.output
    all_imgs = glob.glob(os.path.join(img_folder, "*.jpg"))
    all_imgs.sort()
    os.makedirs(out_dir, exist_ok=True)
    align_sequence(all_imgs, args.autocrop)

    