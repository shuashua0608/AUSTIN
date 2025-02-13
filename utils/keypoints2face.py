import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import random
from PIL import Image
from scipy.optimize import curve_fit
import warnings
from scipy.misc import imsave, imresize, toimage
from tqdm import tqdm
from skimage.transform import resize
from skimage import img_as_bool
import re

add_upper_face = False

part_list = [[list(range(0, 17)) + ((list(range(68, 83)) + [0]) if add_upper_face else [])],  # face
             [range(17, 22)],  # right eyebrow
             [range(22, 27)],  # left eyebrow
             [[28, 31], range(31, 36), [35, 28]],  # nose
             [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
             [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
             [range(48, 55), [54, 55, 56, 57, 58, 59, 48], range(60, 65), [64, 65, 66, 67, 60]],  # mouth and tongue]
             ]


def get_crop_coords(keypoints):
    min_y, max_y = int(keypoints[:, 1].min()), int(keypoints[:, 1].max())
    min_x, max_x = int(keypoints[:, 0].min()), int(keypoints[:, 0].max())
    x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2
    w = h = (max_x - min_x)
    min_x = x_cen - w
    min_y = y_cen - h * 1.25
    max_x = min_x + w * 2
    max_y = min_y + h * 2
    return int(min_y), int(max_y), int(min_x), int(max_x)


def read_keypoints(keypoints):
    if add_upper_face:
        pts = keypoints[:17, :].astype(np.int32)
        baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
        upper_pts = pts[1:-1, :].copy()
        upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
        keypoints = np.vstack((keypoints, upper_pts[::-1, :]))

    return keypoints


def crop(img, coords):
    min_y, max_y, min_x, max_x = coords
    if isinstance(img, np.ndarray):
        return img[min_y:max_y, min_x:max_x]
    else:
        return img.crop((min_x, min_y, max_x, max_y))


def get_face_image(keypoints, size, bw):
    w, h = size
    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8)  # edge map for all edges
    for edge_list in part_list:
        for edge in edge_list:
            for i in range(0, max(1, len(edge) - 1),
                           edge_len - 1):  # divide a long edge into multiple small edges when drawing
                sub_edge = edge[i:i + edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]

                curve_x, curve_y = interp_points(x, y)  # interp keypoints to get the curve shape
                draw_edge(im_edges, curve_x, curve_y, bw=bw)
    return im_edges


# Given the start and end points, interpolate to get a line.
def interp_points(x, y):
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interp_points(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(x) < 3:
                popt, _ = curve_fit(linear, x, y)
            else:
                popt, _ = curve_fit(func, x, y)
                if abs(popt[0]) > 1:
                    return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], (x[-1] - x[0]))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)


def func(x, a, b, c):
    return a * x ** 2 + b * x + c


def linear(x, a, b):
    return a * x + b


# Set colors given a list of x and y coordinates for the edge.
def draw_edge(im, x, y, bw=1, color=(255, 255, 255), draw_end_points=False):
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # edge
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h - 1, y + i))
                xx = np.maximum(0, np.minimum(w - 1, x + j))
                set_color(im, yy, xx, color)

        # edge endpoints
        if draw_end_points:
            for i in range(-bw * 2, bw * 2):
                for j in range(-bw * 2, bw * 2):
                    if (i ** 2) + (j ** 2) < (4 * bw ** 2):
                        yy = np.maximum(0, np.minimum(h - 1, np.array([y[0], y[-1]]) + i))
                        xx = np.maximum(0, np.minimum(w - 1, np.array([x[0], x[-1]]) + j))
                        set_color(im, yy, xx, color)


# Set a pixel to the given color.
def set_color(im, yy, xx, color):
    if len(im.shape) == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
        else:
            im[yy, xx, 0] = ((im[yy, xx, 0].astype(float) + color[0]) / 2).astype(np.uint8)
            im[yy, xx, 1] = ((im[yy, xx, 1].astype(float) + color[1]) / 2).astype(np.uint8)
            im[yy, xx, 2] = ((im[yy, xx, 2].astype(float) + color[2]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]


def get_imgmsk_image(crop_img, crop_msk):
    w, h = crop_img.size
    new_crop_msk = np.zeros((h, w, 3), dtype=np.uint8)
    new_crop_msk[crop_msk != 0] = [0, 255, 0]
    crop_imgmsk = np.float32(crop_img) * 0.5 + np.float32(new_crop_msk) * 0.5
    crop_imgmsk[crop_msk == 0] = np.float32(crop_img)[crop_msk == 0]
    crop_imgmsk = Image.fromarray(crop_imgmsk.astype('uint8'), 'RGB')
    return crop_imgmsk.copy()


if __name__ == "__main__":
    frame_dir_pth = "/data/stroke/Feature/ori_large_frames/"
    save_mask_dir_pth = "/data/stroke/Feature/ori_large_masks/"
    save_imgmask_dir_pth = "/data/stroke/Feature/ori_large_imgmasks/"
    kp_video_dir_pth = "/data/stroke/Feature/ori_large_keypoints/"
    video_list = os.listdir(kp_video_dir_pth)
    video_list.sort()
    for kp_video_name in video_list:
        kp_video_pth = os.path.join(kp_video_dir_pth, kp_video_name)
        cur_msk_dir_pth = os.path.join(save_mask_dir_pth, kp_video_name)
        cur_imgmsk_dir_pth = os.path.join(save_imgmask_dir_pth, kp_video_name)
        if not os.path.exists(cur_msk_dir_pth):
            os.makedirs(cur_msk_dir_pth)
        if not os.path.exists(cur_imgmsk_dir_pth):
            os.makedirs(cur_imgmsk_dir_pth)
        video_index = int(kp_video_name)
        kp_txt_list = os.listdir(kp_video_pth)
        kp_txt_list.sort()
        for kp_txt_name in tqdm(kp_txt_list):
            kp_txt_pth = os.path.join(kp_video_pth, kp_txt_name)
            # read keypoints
            kp = np.loadtxt(kp_txt_pth, delimiter=',')
            bw = 1
            out_kp = read_keypoints(kp)
            frame_name = kp_txt_name.replace("txt", "jpg")
            frame_pth = os.path.join(frame_dir_pth, kp_video_name, frame_name)
            frame = Image.open(frame_pth)
            out_size = frame.size
            out_msk = get_face_image(out_kp, size=out_size, bw=bw)
            out_imgmsk = get_imgmsk_image(frame, out_msk)
            target_size = out_size
            out_img = frame.resize(target_size)
            out_msk = img_as_bool(resize(out_msk, target_size))*255
            out_imgmsk = out_imgmsk.resize(target_size)
            # save results
            name = frame_name.replace("jpg", "png")
            # out_img.save(os.path.join(cur_out_img_dir_pth, name))
            imsave(os.path.join(cur_msk_dir_pth, name), out_msk)
            out_imgmsk.save(os.path.join(cur_imgmsk_dir_pth, name))
