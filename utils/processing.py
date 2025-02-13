# those files related to pre-processing and data cleaning
import os
import os.path as osp
from decord import VideoReader
from decord import cpu
import cv2
import xlwt
import xlrd
import numpy as np
from scipy.misc import imresize, imsave
import shutil
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, recall_score, f1_score
from imblearn.metrics import sensitivity_specificity_support
from sklearn.model_selection import KFold
from imblearn.metrics import sensitivity_specificity_support


# the property of sliced video
def show_property():
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("data")
    video_dir_pth = "/data/stroke/Slices_track_224"
    st_txt_pth = "/data/stroke/RawData/new_stroke.txt"
    nst_txt_pth = "/data/stroke/RawData/new_non_stroke.txt"
    st_txt = open(st_txt_pth, 'r')
    st_files = st_txt.read().splitlines()
    st_files.sort()
    nst_txt = open(nst_txt_pth, 'r')
    nst_files = nst_txt.read().splitlines()
    nst_files.sort()

    st_frame_nums = []
    for i, video_name in enumerate(st_files):
        video_pth = osp.join(video_dir_pth, video_name+'.avi')
        vr = VideoReader(video_pth, ctx=cpu(0))
        print(video_name, len(vr), "stroke")
        sheet1.write(i, 0, video_name)
        sheet1.write(i, 1, str(len(vr)))
        sheet1.write(i, 2, "stroke")
        st_frame_nums.append(len(vr))
    num_st = len(st_files)
    print("min:", min(st_frame_nums))
    print("max:", max(st_frame_nums))
    print("avg:", sum(st_frame_nums)/num_st)
    print("sum:", sum(st_frame_nums))  # 90923

    nst_frame_nums = []
    for i, video_name in enumerate(nst_files):
        video_pth = osp.join(video_dir_pth, video_name + '.avi')
        vr = VideoReader(video_pth, ctx=cpu(0))
        print(video_name, len(vr), "nonstroke")
        sheet1.write(i+num_st, 0, video_name)
        sheet1.write(i+num_st, 1, str(len(vr)))
        sheet1.write(i+num_st, 2, "nonstroke")
        nst_frame_nums.append(len(vr))
    num_nst = len(nst_files)
    print("min:", min(nst_frame_nums))
    print("max:", max(nst_frame_nums))
    print("avg:", sum(nst_frame_nums)/num_nst)
    print("sum:", sum(nst_frame_nums))  # 44879
    # workbook.save("/data/stroke/stroke_data.xls")


# generate frames
def generate_frames():
    # frame_dir = "/data/stroke/Feature/Frames256"
    # video_dir_pth = "/data/stroke/Slices_track_224"
    frame_dir = "/data/stroke/Feature/Ori_Frames256"
    if not osp.exists(frame_dir):
        os.makedirs(frame_dir)
    video_dir_pth = "/data/stroke/original_video"
    st_txt_pth = "/data/stroke/RawData/stroke_1.txt"
    nst_txt_pth = "/data/stroke/RawData/nonstroke_1.txt"
    st_txt = open(st_txt_pth, 'r')
    st_files = st_txt.read().splitlines()
    st_files.sort()
    nst_txt = open(nst_txt_pth, 'r')
    nst_files = nst_txt.read().splitlines()
    nst_files.sort()

    stroke_frame_dir_pth = osp.join(frame_dir, "stroke")
    nonstroke_frame_dir_pth = osp.join(frame_dir, "nonstroke")

    for i, video_name in enumerate(st_files):
        video_pth = osp.join(video_dir_pth, video_name+'.MOV')
        vr = VideoReader(video_pth, ctx=cpu(0))
        num_frame = len(vr)
        assert num_frame > 0
        cur_video_dir = osp.join(stroke_frame_dir_pth, video_name)
        print(cur_video_dir)
        if not osp.exists(cur_video_dir):
            os.makedirs(cur_video_dir)

        for j in range(num_frame):
            frame = vr[j].asnumpy()
            frame = imresize(frame, (256, 256, 3))
            frame_name = osp.join(cur_video_dir, format(j, "04d")+".jpg")
            imsave(frame_name, frame)

    for i, video_name in enumerate(nst_files):
        video_pth = osp.join(video_dir_pth, video_name+'.MOV')
        vr = VideoReader(video_pth, ctx=cpu(0))
        num_frame = len(vr)
        assert num_frame > 0
        cur_video_dir = osp.join(nonstroke_frame_dir_pth, video_name)
        print(cur_video_dir)
        if not osp.exists(cur_video_dir):
            os.makedirs(cur_video_dir)

        for j in range(num_frame):
            frame = vr[j].asnumpy()
            frame = imresize(frame, (256, 256, 3))
            frame_name = osp.join(cur_video_dir, format(j, "04d")+".jpg")
            imsave(frame_name, frame)


# process large
# only get the first 300 frames
# and then sample 60 frames using Uniform Sampling
def trunc_large_frame():
    video_dir_pth = "/data/stroke/large"
    save_dir_pth = "/data/stroke/Feature/ori_large_frames"
    video_dir_list = os.listdir(video_dir_pth)
    video_dir_list.sort()
    # video_dir_list = ["0003.avi"]
    for video_name in video_dir_list[117:]:
        print(video_name)
        video_pth = os.path.join(video_dir_pth, video_name)
        vr = cv2.VideoCapture(video_pth)
        num_frames = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_frames <= 300:
            print(video_name, num_frames)
            continue
        interval = (300 - 1) // (60 - 1)
        count = 0
        frame_id = 0
        success, image = vr.read()
        frame_dir_pth = os.path.join(save_dir_pth, video_name[:-4])
        if not os.path.exists(frame_dir_pth):
            os.makedirs(frame_dir_pth)
        while success:
            if count % interval == 0:
                filename = "%03d_frame_%03d.jpg" % (int(video_name[:-4]), frame_id)
                image = cv2.resize(image, (256, 256))
                cv2.imwrite(os.path.join(frame_dir_pth, filename), image)
                frame_id += 1
                if frame_id == 60:
                    break
            success, image = vr.read()
            count += 1
        vr.release()


# analyze the statistics of large
def analyze_large():
    stroke_txt = "/data/stroke/RawData/stroke_1.txt"
    non_stroke_txt = "/data/stroke/RawData/nonstroke_2.txt"
    video_dir_pth = "/data/stroke/Feature/ori_large_imgmasks"
    stroke_list = open(stroke_txt, "r").read().splitlines()
    stroke_list.sort()
    non_stroke_list = open(non_stroke_txt, "r").read().splitlines()
    non_stroke_list.sort()
    for video_name in stroke_list:
        video_pth = os.path.join(video_dir_pth, video_name)
        frame_list = os.listdir(video_pth)
        n_frames = len(frame_list)
        if n_frames != 60:
            print("stroke", video_name, n_frames)
    for video_name in non_stroke_list:
        video_pth = os.path.join(video_dir_pth, video_name)
        frame_list = os.listdir(video_pth)
        n_frames = len(frame_list)
        if n_frames != 60:
            print("non-stroke", video_name, n_frames)


# analyze the statistics of clinician
def analyze_clinician():
    xlsx_data = "/data/stroke/Stroke_GT.xlsx"
    workbook = xlrd.open_workbook(xlsx_data)
    worksheet = workbook.sheet_by_index(0)
    col_data = []
    for i in range(worksheet.ncols):
        col_data.append(worksheet.col_values(i))
    stroke_txt_pth = "/data/stroke/RawData/stroke_1.txt"
    non_stroke_txt_pth = "/data/stroke/RawData/nonstroke_1.txt"
    stroke_txt = open(stroke_txt_pth, "r").readlines()
    stroke_name_list = []
    for line in stroke_txt:
        stroke_name_list.append(line.replace("\n", ""))
    non_stroke_txt = open(non_stroke_txt_pth, "r").readlines()
    non_stroke_name_list = []
    for line in non_stroke_txt:
        non_stroke_name_list.append(line.replace("\n", ""))
    non_stroke_name_list = non_stroke_name_list[:-3]
    dataset_name_list = stroke_name_list + non_stroke_name_list
    dataset_name_list.sort()
    xlsx_name_list = np.array(col_data[0][1:], dtype=str)
    xy, x_ind, y_ind = np.intersect1d(xlsx_name_list, dataset_name_list, return_indices=True)
    TOTAL_NAMES = xy
    # calculate metric
    TOTAL_TARGETS = np.array(col_data[1][1:])[x_ind]
    TOTAL_PREDS = np.array(col_data[2][1:])[x_ind]
    # add 0001 and 0002
    TOTAL_NAMES = np.insert(TOTAL_NAMES, 0, '0001')
    TOTAL_NAMES = np.insert(TOTAL_NAMES, 1, '0002')
    TOTAL_TARGETS = np.insert(TOTAL_TARGETS, 0, 0.0)
    TOTAL_TARGETS = np.insert(TOTAL_TARGETS, 1, 0.0)
    TOTAL_PREDS = np.insert(TOTAL_PREDS, 0, 0.0)
    TOTAL_PREDS = np.insert(TOTAL_PREDS, 1, 0.0)
    sensitivity, specificity, _ = sensitivity_specificity_support(TOTAL_TARGETS, TOTAL_PREDS)
    print("Accuracy: ", str(accuracy_score(TOTAL_TARGETS, TOTAL_PREDS)))
    print("Precision: ", str(precision_score(TOTAL_TARGETS, TOTAL_PREDS)))
    print("Recall: ", str(recall_score(TOTAL_TARGETS, TOTAL_PREDS)))
    print("F1:", str(f1_score(TOTAL_TARGETS, TOTAL_PREDS)))
    print("AUC: ", str(roc_auc_score(TOTAL_TARGETS, TOTAL_PREDS)))
    print("Sensitivity:", sensitivity[1])
    print("Specificity:", specificity[1])
    # correct label
    # new_stroke_txt_pth = "/data/stroke/RawData/new_stroke.txt"
    # new_non_stroke_txt_pth = "/data/stroke/RawData/new_non_stroke.txt"
    # new_stroke_txt = open(new_stroke_txt_pth, "w")
    # new_non_stroke_txt = open(new_non_stroke_txt_pth, "w")
    # for ind, name in enumerate(TOTAL_NAMES):
    #     label = TOTAL_TARGETS[ind]
    #     if label:
    #         new_stroke_txt.writelines(name+'\n')
    #     else:
    #         new_non_stroke_txt.writelines(name+'\n')
    # new_stroke_txt.close()
    # new_non_stroke_txt.close()
    # new_stroke_txt = open(new_stroke_txt_pth, "r")
    # new_non_stroke_txt = open(new_non_stroke_txt_pth, "r")
    # print("#stroke:", len(new_stroke_txt.readlines()))
    # print("#non_stroke:", len(new_non_stroke_txt.readlines()))


def compute_mean_std_for_cv_clinician():
    xlsx_data = "/data/stroke/Stroke_GT.xlsx"
    workbook = xlrd.open_workbook(xlsx_data)
    worksheet = workbook.sheet_by_index(0)
    col_data = []
    for i in range(worksheet.ncols):
        col_data.append(worksheet.col_values(i))
    stroke_txt_pth = "/data/stroke/RawData/new_stroke.txt"
    non_stroke_txt_pth = "/data/stroke/RawData/new_non_stroke.txt"
    stroke_txt = open(stroke_txt_pth, "r").readlines()
    stroke_name_list = []
    for line in stroke_txt:
        stroke_name_list.append(line.replace("\n", ""))
    non_stroke_txt = open(non_stroke_txt_pth, "r").readlines()
    non_stroke_name_list = []
    for line in non_stroke_txt:
        non_stroke_name_list.append(line.replace("\n", ""))
    dataset_name_list = stroke_name_list + non_stroke_name_list
    dataset_name_list.sort()
    xlsx_name_list = np.array(col_data[0][1:], dtype=str)
    xy, x_ind, y_ind = np.intersect1d(xlsx_name_list, dataset_name_list, return_indices=True)
    TOTAL_NAMES = xy
    # calculate metric
    TOTAL_TARGETS = np.array(col_data[1][1:])[x_ind]
    TOTAL_PREDS = np.array(col_data[2][1:])[x_ind]
    # add 0001 and 0002
    TOTAL_NAMES = np.insert(TOTAL_NAMES, 0, '0001')
    TOTAL_NAMES = np.insert(TOTAL_NAMES, 1, '0002')
    TOTAL_TARGETS = np.insert(TOTAL_TARGETS, 0, 0.0)
    TOTAL_TARGETS = np.insert(TOTAL_TARGETS, 1, 0.0)
    TOTAL_PREDS = np.insert(TOTAL_PREDS, 0, 0.0)
    TOTAL_PREDS = np.insert(TOTAL_PREDS, 1, 0.0)
    clinician_dict = {}
    for i, name in enumerate(TOTAL_NAMES):
        clinician_dict[name] = [TOTAL_PREDS[i], TOTAL_TARGETS[i]]
    totallist = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    for train1, test1 in kf.split(stroke_name_list):
        totallist.append([stroke_name_list[x] for x in test1])

    for train1, test1 in kf.split(non_stroke_name_list):
        totallist.append([non_stroke_name_list[x] for x in test1])

    spec_res = np.zeros(5)
    sens_res = np.zeros(5)
    for i in range(5):
        stroke_list = totallist[i]
        non_stroke_list = totallist[i + 5]
        preds = [clinician_dict[name][0] for name in stroke_list+non_stroke_list]
        targets = [clinician_dict[name][1] for name in stroke_list+non_stroke_list]
        temp_targets = [1]*len(stroke_list) + [0]*len(non_stroke_list)
        assert temp_targets == targets
        sensitivity, specificity, _ = sensitivity_specificity_support(targets, preds)
        spec_res[i] = specificity[1]
        sens_res[i] = sensitivity[1]
    print("specificity:")
    print(spec_res)
    print(np.mean(spec_res), np.std(spec_res))
    print("sensitivity:")
    print(sens_res)
    print(np.mean(sens_res), np.std(sens_res))


# correct directory
def correct():
    stroke_dir = "/data/stroke/Feature/Frames256/stroke"
    non_stroke_dir = "/data/stroke/Feature/Frames256/nonstroke"
    stroke_txt_pth = "/data/stroke/RawData/new_stroke.txt"
    non_stroke_txt_pth = "/data/stroke/RawData/new_non_stroke.txt"
    stroke_txt = open(stroke_txt_pth, "r")
    non_stroke_txt = open(non_stroke_txt_pth, "r")
    stroke_list = stroke_txt.read().splitlines()
    non_stroke_list = non_stroke_txt.read().splitlines()
    for name in stroke_list:
        stroke_file = os.path.join(stroke_dir, name)
        if not os.path.exists(stroke_file):
            print(stroke_file)
            old_stroke_file = os.path.join(non_stroke_dir, name)
            assert os.path.exists(old_stroke_file)
            shutil.move(old_stroke_file, stroke_file)

    for name in non_stroke_list:
        non_stroke_file = os.path.join(non_stroke_dir, name)
        if not os.path.exists(non_stroke_file):
            print(non_stroke_file)
            old_non_stroke_file = os.path.join(stroke_dir, name)
            assert os.path.exists(old_non_stroke_file)
            shutil.move(old_non_stroke_file, non_stroke_file)


def count_frame_num():
    root_dir = "/data/stroke/Feature/NewFrames256"
    video_list = os.listdir(root_dir)
    video_list.sort()
    for video_name in video_list:
        video_pth = os.path.join(root_dir, video_name)
        frame_list = os.listdir(video_pth)
        print(video_name, len(frame_list))


def compute_mean_std_for_cv():
    root_dir = "/data/stroke/video-snapshots/snapshots-jump-W2.00/results"
    names = np.load(os.path.join(root_dir, "name.npy"))
    pred = np.load(os.path.join(root_dir, "pred.npy"))
    score = np.load(os.path.join(root_dir, "score.npy"))
    target = np.load(os.path.join(root_dir, "target.npy"))
    strokes = []
    nonstrokes = []
    stroke_txt = "/data/stroke/RawData/new_stroke.txt"  # stroke.txt
    nonstroke_txt = "/data/stroke/RawData/new_non_stroke.txt"  # nonstroke.txt
    with open(stroke_txt, 'r') as f:
        for vids in f.read().splitlines():
            strokes.append(vids)
    with open(nonstroke_txt, "r") as f:
        for vids in f.read().splitlines():
            nonstrokes.append(vids)
    totallist = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    for train1, test1 in kf.split(strokes):
        totallist.append([strokes[x] for x in test1])

    for train1, test1 in kf.split(nonstrokes):
        totallist.append([nonstrokes[x] for x in test1])

    sample_nums = np.zeros(5)
    for i in range(5):
        stroke_num = len(totallist[i])
        non_stroke_num = len(totallist[i+5])
        sample_nums[i] = stroke_num + non_stroke_num

    split_inds = np.zeros(6, dtype=np.int)
    for i in range(5):
        split_inds[i+1] = sample_nums[i] + split_inds[i]

    spec_res = np.zeros(5)
    sens_res = np.zeros(5)
    acc_res = np.zeros(5)
    auc_res = np.zeros(5)
    for i in range(5):
        cur_score = score[split_inds[i]:split_inds[i+1]]
        cur_pred = pred[split_inds[i]:split_inds[i+1]]
        cur_target = target[split_inds[i]:split_inds[i+1]]
        sensitivity, specificity, _ = sensitivity_specificity_support(cur_target.tolist(), cur_pred.tolist())
        spec_res[i] = specificity[1]
        sens_res[i] = sensitivity[1]
        acc_res[i] = accuracy_score(cur_target, cur_pred)
        auc_res[i] = roc_auc_score(cur_target, cur_score)
    print(root_dir)
    print("specificity:")
    print(spec_res)
    print(np.mean(spec_res), np.std(spec_res))
    print("sensitivity:")
    print(sens_res)
    print(np.mean(sens_res), np.std(sens_res))
    print("ACC")
    print(acc_res)
    print(np.mean(acc_res), np.std(acc_res))
    print("AUC")
    print(auc_res)
    print(np.mean(auc_res), np.std(auc_res))


# calculate clip property
def cal_clip_property():
    dir_pth = "/data/stroke/frames"
    dirs = os.listdir(dir_pth)
    dirs.sort()
    dirs = dirs[:122]
    for video_name in dirs:
        video_pth = os.path.join(dir_pth, video_name)
        frame_list = os.listdir(video_pth)
        frame_len = len(frame_list)
        if frame_len == 0:
            continue
        clip_num = 0
        clip_len = 1
        clip_prop = []
        start_idx = [frame_list[0][:-4]]
        end_idx = []
        for i in range(frame_len-1):
            ind1 = int(frame_list[i][:-4])
            ind2 = int(frame_list[i + 1][:-4])
            if ind2 - ind1 != 1:
                end_idx.append(ind1)
                start_idx.append(ind2)
                clip_num += 1
                clip_prop.append(clip_len)
                clip_len = 1
            else:
                clip_len += 1
        start_idx = start_idx[:-1]
        # print(video_name, max(clip_prop), min(clip_prop), len(clip_prop), sum(clip_prop)/len(clip_prop))
        print(video_name, max(clip_prop))


def analyse_sex_info():
    xlsx_data = "/data/stroke/STROKE_METADATA.xlsx"
    workbook = xlrd.open_workbook(xlsx_data)
    worksheet = workbook.sheet_by_index(0)
    col_data = []
    for i in range(worksheet.ncols):
        col_data.append(worksheet.col_values(i))
    name = col_data[0]
    sex = col_data[1]
    print()


def add_new_videos():
    extra_folder_pth = "/data/stroke/upload/extra_specs"
    new_folder_pth = "/data/stroke/spec"
    extra_folder_list = os.listdir(extra_folder_pth)
    extra_folder_list.sort()
    if ".DS_Store" in extra_folder_list:
        extra_folder_list.remove(".DS_Store")
    for name in extra_folder_list:
        print(name)
        extra_folder = os.path.join(extra_folder_pth, name)
        new_folder = os.path.join(new_folder_pth, name)
        if os.path.isdir(extra_folder):
            shutil.copytree(extra_folder, new_folder)
        else:
            shutil.copyfile(extra_folder, new_folder)


if __name__ == "__main__":
    add_new_videos()


# /data/stroke/video-snapshots/snapshots-jump-adv-deep-reverse-W2.00/results
# specificity:
# [0.28571429 0.42857143 0.28571429 0.71428571 0.42857143]
# 0.42857142857142855 0.15649215928719035
# sensitivity:
# [0.875      0.86666667 1.         0.73333333 0.93333333]
# 0.8816666666666666 0.08825468196582487
# ACC
# [0.69565217 0.72727273 0.77272727 0.72727273 0.77272727]
# 0.7391304347826086 0.029762609639552087
# AUC
# [0.58928571 0.59047619 0.63809524 0.75238095 0.74285714]
# 0.6626190476190477 0.07166468910418221

# /data/stroke/video-snapshots/snapshots-jump-W2.00/results
# specificity:
# [0.14285714 0.         0.28571429 0.28571429 0.28571429]
# 0.19999999999999998 0.11428571428571428
# sensitivity:
# [0.8125     1.         0.86666667 0.73333333 0.86666667]
# 0.8558333333333333 0.08706638591072652
# ACC
# [0.60869565 0.68181818 0.68181818 0.59090909 0.68181818]
# 0.6490118577075098 0.040571153898359934
# AUC
# [0.51785714 0.66666667 0.59047619 0.52380952 0.53333333]
# 0.5664285714285715 0.056385858859869124
