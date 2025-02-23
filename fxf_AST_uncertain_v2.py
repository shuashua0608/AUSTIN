# MMDL with addressing jumping frames
import os
os.putenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["WORLD_SIZE"] = "1"
import sys
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import madgrad
import adabound
import torch.nn as nn
import torch.nn.parallel
import pandas as pd
import torch.optim
import torch.utils.data
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, recall_score, f1_score, roc_curve
from utils.misc import my_logger
import dataset.load_S4_MedIA_AST_ADV_SEG_uncertain as dataload 
import model.S4_facexformer_AST_ADV_uncertain as my_models
from utils import model_util
from imblearn.metrics import sensitivity_specificity_support
import glob
from torchvision import models
import torch.optim as optim# MMDL with addressing jumping frames

from torch.utils.tensorboard import SummaryWriter
import wandb

NUM_EPOCH = 30
LEANRING_RATE = 1e-3 #last one: 1e-3
CV_FOLD = 5
LAMBDA = 5
# BATCH_SIZE_TRAIN = 64
# BATCH_SIZE_TEST = 64
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 16
# root = "/data/stroke"
root = "."
wi = [2.0, 1.0]   # [1.25, 1.0]
w_cal=True
w_weight =True

parser = argparse.ArgumentParser(description='MMDL')
parser.add_argument('--lr-ratio', '-r', default=2, type=float,
                    metavar='LRR', help='Ratio of losses')
parser.add_argument('--lamb', '-l', default=LAMBDA, type=int,
                    metavar='L', help='Ratio of losses')
parser.add_argument('--w', default=w_cal, type=bool)
parser.add_argument('--wi', default=w_weight, type=bool)
parser.add_argument('--epochs', default=NUM_EPOCH, type=int, 
                    help='number of total epochs to run')
parser.add_argument('--version', default=5, type=int, 
                    help='data version')
parser.add_argument('--step', default=999, type=int, 
                    help='step optimizer')
parser.add_argument('--lr', '--learning-rate', default=LEANRING_RATE, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--patience', default=10, type=float, help='Patience for learning rate scheduler')
parser.add_argument('--print-freq', '-p', default=300, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--transformer', action='store_true')
parser.add_argument('--clip', action='store_true')
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--param', action='store_true')
# parser.add_argument('--grad', action='store_true')
parser.add_argument('--adv', action='store_true')
parser.add_argument('--all-patient', action='store_true')

args = parser.parse_args()

postfix = "NEW-fxf-AST-UNCERTAIN-E%d-ADAMW-1E-3-w-%s-weight-%s-softplus"%(args.epochs, args.w, args.wi)
postfix += "-step%d"%args.step if args.step != 999 else "-NOSTEP"
if args.norm: postfix += "-NORM"
if args.transformer: postfix += "-TRANS"
if args.clip: postfix += "-CLIP"
if args.param: postfix += "-PARAM"
if args.cosine: postfix += "-COS"
if args.all_patient: postfix += "-all-patient"
# if args.grad: postfix += "-GRAD"
if args.adv: postfix += "-ADV"
load_flag = "-load" in postfix  # load means that we directly use the trained model
restore_from = "video-snapshots-deterministic/snapshots-s4-W%.2f-L200-freeze/ckpt_10" % (wi[0]) 
# large means face video with edge
device = "cuda"
writer = SummaryWriter('runs/log'+postfix)

    
SNAPSHOT_DIR = root + '/video-snapshots-deterministic-64x7-fix-grad/snapshots' +postfix
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE_TRAIN, "04d")+"E"+format(args.epochs, "04d")+".log"
if not load_flag:
    sys.stdout = my_logger(LOG_PATH, sys.stdout)


class UncertaintyLoss(nn.Module):
    def __init__(self, alpha = 1.0,  epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.w_cal = w_cal
        self.w_weight = w_weight

    def forward(self, y_hat, y_mri, y_triage, sigma):
        """
        y_hat: Raw logits, shape (batch_size, 2)
        y_mri: Ground truth MRI labels (0 or 1), shape (batch_size,)
        y_triage: Triage labels (0 or 1), shape (batch_size,)
        sigma: Estimated uncertainty, shape (batch_size, 1)
        """
        # Ensure sigma is positive
        sigma = torch.abs(sigma) + self.epsilon

        if self.w_weight:
            criterion = nn.CrossEntropyLoss(torch.tensor(wi)).cuda() #weight=torch.tensor(wi)
        else:
            criterion = nn.CrossEntropyLoss().cuda() 
            
        ce_loss = criterion(y_hat, y_mri)
        if self.w_cal:
            w = torch.exp(-self.alpha * torch.abs(y_mri - y_triage))
        else:
            w = 0
        # Uncertainty loss component
        loss = ce_loss / (2 * sigma**2) + w * torch.log(sigma)

        return loss.mean()


def examine_thre(thres,score,target):
    new_thre = score.copy()
    new_thre[new_thre<thres]=0
    new_thre[new_thre>=thres]=1
    sensitivity, specificity, _ = sensitivity_specificity_support(target, new_thre)
    print("Accuracy: ", str(accuracy_score(target, new_thre)))
    print("Specificity:", specificity[1])
    print("Sensitivity:", sensitivity[1])
    print("AUC: ", str(roc_auc_score(target, score)))
    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def best_thre(score,target,expected_specificty=None):
    fpr, tpr, thresholds = roc_curve(target, score)
    gmeans = np.sqrt(tpr * (1-fpr))
    if not expected_specificty:
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        return thresholds[ix]
    else:
        tgt_fpr = 1 - expected_specificty
        ix = find_nearest(fpr,tgt_fpr)
        return thresholds[ix]
        
def setup_optimizer(model, lr, weight_decay, patience):
    """
    S4 requires a specific optimizer setup.
    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.
    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(
    # optimizer = madgrad.MADGRAD(
        params,
        lr=lr,
        weight_decay=1e-4
    )
    
    # optimizer_step = adabound.AdaBound(
    optimizer_step = torch.optim.SGD(
    # optimizer_step = torch.optim.AdamW(
        params,
        lr=1e-4,
        # lr=lr,
        # momentum=0.9
        weight_decay=1e-4
        )
    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in set(frozenset(hp.items()) for hp in hps)
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        # hp['lr'] = 3e-6
        hp['lr'] = 1e-6
        optimizer.add_param_group(
            {"params": params, **hp}
        )
        optimizer_step.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer_step, step_size=30, gamma=0.5)
    # # Print optimizer info
    # keys = sorted(set([k for hp in hps for k in hp.keys()]))
    # for i, g in enumerate(optimizer.param_groups):
    #     group_hps = {k: g.get(k, None) for k in keys}
    #     print(' | '.join([
    #                          f"Optimizer group {i}",
    #                          f"{len(g['params'])} tensors",
    #                      ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, optimizer_step

def main():
    global args, best_prec1
    # wandb.init(
    #     # Set the project where this run will be logged
    #     project="M3Stroke-Custom-S4", 
    #     # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    #     name="fix-grad-experiment"+postfix, 
    #     # Track hyperparameters and run metadata
    #     config={
    #     "learning_rate": LEANRING_RATE,
    #     "epochs": args.epochs,
    #     }
    #     )
    total_end = time.time()
    NUM_CLASS = 2
    random_seed = 123
    strokes = []
    nonstrokes = []
    arg_fold = CV_FOLD
    arg_batchsize_train = BATCH_SIZE_TRAIN
    arg_batchsize_eval = BATCH_SIZE_TEST
    arg_root = '../stroke_data/video_segment_fix/'  # '/Feature/Frames256/', /Feature/ori_large_frames/
    # audio_root = '../Audio/new_trim/16k/cookie/'
    audio_root = '../stroke_data/fix_len_aud/segment/'
    # audio_root = '../Stroke_data/7seg/audio_segment_%dk/'%args.sr
    spec_path = '../stroke_data/fix_len_aud/spec/'
    #audio_root = '../Audio/segment/'
    #spec_path = '../Audio/Spectrograms/'
    # stroke_txt = root + "/RawData/new_stroke.txt"   # stroke.txt
    # nonstroke_txt = root + "/RawData/new_non_stroke.txt"  # nonstroke.txt
    print(arg_root)
    # print(stroke_txt)
    # print(nonstroke_txt)
    print("train bs:", arg_batchsize_train)
    print("val bs:", arg_batchsize_eval)
    print("seed:", random_seed)
    # torch.use_deterministic_algorithms(True)
    setup_seed(random_seed)
    print("loss weight:", wi)
    print("EPOCH:", NUM_EPOCH)
    print("postfix:", postfix)
    print("load flag:", load_flag)
    print("restore from:", restore_from)

    
    df = pd.read_csv('../triage_gt.csv', dtype=object)
    df.ID = df.ID.apply(lambda x: "%04d"%int(x))

    strokes = []
    nonstrokes = []
    tri_dic = {}
    
    for _, row in df.iterrows():
        tri_tmp=row['Triage']
        gt_tmp = row['GT']
        if row['ID'] in ['0018','0147','0193','0119','0259','0274','0188','0058','0036','0078','0283','0239']:
            continue
        if (gt_tmp=='0') & (pd.notnull(tri_tmp)):
            nonstrokes.append(row['ID'])
            tri_dic[row['ID']]=tri_tmp
        if (gt_tmp=='1') & (pd.notnull(tri_tmp)):
            strokes.append(row['ID'])
            tri_dic[row['ID']]=tri_tmp

    print("num of stroke:", len(strokes))
    print("num of nonstroke:", len(nonstrokes))

    totallist = []
    totallist.append([j for j in strokes if int(j) < 196])
    totallist.append([j for j in strokes if (int(j) >= 196 and int(j) < 237)])
    totallist.append([j for j in strokes if int(j) >= 237 and int(j) < 296])
    totallist.append([j for j in nonstrokes if int(j) < 196])
    totallist.append([j for j in nonstrokes if (int(j) >= 196 and int(j) < 237)])
    totallist.append([j for j in nonstrokes if int(j) >= 237 and int(j) < 296])

    ans = []
    TOTAL_PREDS = []
    TOTAL_TARGETS = []
    TOTAL_SCORES = []
    TOTAL_NAMES = []
    best_auc = 0
    best_prec1 = 0

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = dataload.LoadData(arg_root, audio_root,     spec_path, 0, totallist, tri_dic, arg_batchsize_train,arg_batchsize_eval)
    args.class_num = NUM_CLASS
    ResNet_parameterDir = 'model/fair_7.pt'
    model = my_models.S4_fxf_AST(video_model_pth=ResNet_parameterDir, n_layers=3, d_model=3072,d_input=1024,d_output=2,dropout=0.4,l_max=64,
            afternorm=args.norm,
            fparam=args.param,
            Trans=args.transformer)
    # model.encoder.audio_model = torch.load('model/resnet18_aud.pth')
    dis = my_models.Discriminator(n_downsampling=1, ndf=36)

    print(model)
    for param in model.parameters():
        param.requires_grad = True
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.ast.parameters():
        param.requires_grad = False

    model.cuda()
    dis.cuda()
    gradient_scaler = torch.cuda.amp.GradScaler()

    criterion1 = UncertaintyLoss(alpha=1.0) # nn.CrossEntropyLoss(weight=torch.tensor(wi)).cuda()
    criterion2 = nn.MSELoss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
    #                               betas=(0.5, 0.999))
    
    optimizer,optimizer_step = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience
    )
    optimizer_adv = torch.optim.Adam(filter(lambda p: p.requires_grad, dis.parameters()), 1e-6,
                                  betas=(0.5, 0.999))
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10, eta_min=1e-4) if args.cosine else None
    # scheduler_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_step,T_max=len(train_loader), eta_min=1e-5)
    
    cudnn.benchmark = False
    best_pred = None
    sigma_val_best = None
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        if args.cosine:
            scheduler.step()
        if epoch == args.step:
            optimizer = optimizer_step
            # scheduler = scheduler2
        print("=====Training=====")
        #preds, pred_matrix_fc.cpu()[:, 1], target_vector.cpu()
        skip_lr_sched = train(gradient_scaler, train_loader,train_dataset, model, dis, criterion1, criterion2, optimizer, optimizer_adv, epoch) # predtmp, predscore, targ
        #auc = roc_auc_score(targ.data.tolist(),predscore.data.tolist())
        #writer.add_scalar('AUC/test', auc, epoch)
        print("=====Validating=====")
        prec1, predtmp, predscore, targ, result_names, _ = validate(val_loader, model, val_dataset)
        # model_util.adjust_learning_rate(optimizer, epoch, args.lr, args.epochs)
        auc = roc_auc_score(targ.data.tolist(),predscore.data.tolist())
        writer.add_scalar('AUC/val', auc, epoch)
        # add testing monitoring
        _, _, predscore_test, targ_test, _ , sigma_val = validate(test_loader, model, test_dataset)
        auc_test = roc_auc_score(targ_test.data.tolist(),predscore_test.data.tolist())
        writer.add_scalar('AUC/test', auc_test, epoch)
        # wandb.log({"AUC/test": auc_test})
        is_best = auc > best_auc
        # is_best = prec1 > best_prec1
        
        if epoch == 20:
            is_best = True
            best_auc = auc
            
        if is_best:
            best_pred = predtmp
            best_targ = targ
            best_score = predscore
            print(best_targ.data.tolist(), best_pred.data.tolist()[0])
            print('better model!')
            best_auc = max(auc, best_auc)
            # best_prec1 = max(prec1, best_prec1)
            # delete the former model
            model_name = 'ST_B' + format(BATCH_SIZE_TRAIN, "04d") + '_E*'
            model_list = glob.glob(os.path.join(SNAPSHOT_DIR, model_name))
            for model_file in model_list:
                try:
                    os.remove(model_file)
                except:
                    print("Error while deleting file:", model_file)
            # save the best model
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(SNAPSHOT_DIR,
                                    'ST_B' + format(BATCH_SIZE_TRAIN, "04d")
                                    + '_E' + format(epoch, "06d") + '.pth'))
            sigma_val_best = sigma_val
        else:
            print('Model too bad & not save')
        cur_lr = optimizer.param_groups[0]['lr']
        # if scheduler and not skip_lr_sched:
            
        #     cur_lr = optimizer.param_groups[0]['lr']
        #     optimizer_adv.param_groups[0]['lr'] = cur_lr/args.lr_ratio
        writer.add_scalar('LR', cur_lr, epoch)
        # wandb.log({"LR": cur_lr})
        writer.flush()
    TOTAL_PREDS += best_pred.data.tolist()[0]
    TOTAL_TARGETS += best_targ.data.tolist()
    TOTAL_SCORES += best_score.data.tolist()
    TOTAL_NAMES += result_names
    ans.append(best_prec1)

    thres = best_thre(TOTAL_SCORES, TOTAL_TARGETS)

    print('==========Validation Results==========')
    examine_thre(thres,np.array(TOTAL_SCORES),np.array(TOTAL_TARGETS))
    pred_name = "pred.npy"
    target_name = "target.npy"
    score_name = "score.npy"
    name = "name.npy"
    res_dir = os.path.join(SNAPSHOT_DIR, "results/val/")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    np.save(os.path.join(res_dir, pred_name), np.array(TOTAL_PREDS))
    np.save(os.path.join(res_dir, target_name), np.array(TOTAL_TARGETS))
    np.save(os.path.join(res_dir, score_name), np.array(TOTAL_SCORES))
    np.save(os.path.join(res_dir, name), np.array(TOTAL_NAMES))
    np.save(os.path.join(res_dir, 'sigma_test.npy'), sigma_val_best)
    
    print('==========Testing Results==========')
    model_name = 'ST_B' + format(BATCH_SIZE_TRAIN, "04d") + '_E*'
    model_list = glob.glob(os.path.join(SNAPSHOT_DIR, model_name))
    model_pth = model_list[0]
    ckpt = torch.load(model_pth)
    model.load_state_dict(ckpt["state_dict"])
    
    prec1, predtmp, predscore, targ, result_names, sigma_test = validate(test_loader, model, test_dataset)
    
    best_pred = predtmp
    best_targ = targ
    best_score = predscore
    print(best_targ.data.tolist(), best_pred.data.tolist()[0])
    best_prec1 = max(prec1, best_prec1)
    
    
    
    TOTAL_PREDS = best_pred.data.tolist()[0]
    TOTAL_TARGETS = best_targ.data.tolist()
    TOTAL_SCORES = best_score.data.tolist()
    TOTAL_NAMES = result_names
    
    examine_thre(thres,np.array(TOTAL_SCORES),np.array(TOTAL_TARGETS))
    pred_name = "pred.npy"
    target_name = "target.npy"
    score_name = "score.npy"
    name = "name.npy"
    res_dir = os.path.join(SNAPSHOT_DIR, "results/test")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    np.save(os.path.join(res_dir, pred_name), np.array(TOTAL_PREDS))
    np.save(os.path.join(res_dir, target_name), np.array(TOTAL_TARGETS))
    np.save(os.path.join(res_dir, score_name), np.array(TOTAL_SCORES))
    np.save(os.path.join(res_dir, name), np.array(TOTAL_NAMES))
    
    np.save(os.path.join(res_dir, 'sigma_test.npy'), sigma_test)
    # wandb.finish()

def train(gradient_scaler, train_loader, train_dataset, model, dis, criterion1, criterion2, optimizer1, optimizer2, epoch):
    batch_time = model_util.AverageMeter()
    model_losses = model_util.AverageMeter()
    uncertainty_losses = model_util.AverageMeter()
    fake_dis_losses = model_util.AverageMeter()
    dis_losses = model_util.AverageMeter()
    frameacc = model_util.AverageMeter()
    VideoAcc = model_util.AverageMeter()

    # switch to train mode
    output_store_fc = []
    target_store = []
    index_vector = []
    #TODO: track sigma #
    sigma_vector = [] 
    
    model.train()
    end = time.time()
    model.zero_grad()
    dis.zero_grad()
    for i, (input_first, spec_img, fbank_cookie, target_first, triage_first, index, input_adv, spec_adv, fbank_adv, target_adv) in enumerate(train_loader):
        target_var = target_first.cuda()
        triage_var = triage_first.cuda()
        imgvar = input_first.cuda()
        target_adv = target_adv.to(torch.float16).cuda()
        input_adv = input_adv.cuda()
        cookievar = fbank_cookie.cuda()
        cookieadv = fbank_adv.cuda()
        # print(imgvar.shape, specvar.shape, cookievar.shape)
        optimizer1.zero_grad()
        model.zero_grad()
        dis.zero_grad()
        with torch.cuda.amp.autocast():

            pred_score, pred_logit, sigma, fea = model(imgvar, cookievar, phrase='train')
            fea_adv = model(input_adv, cookieadv, phrase='adv')

            output_store_fc.append(pred_score)
            target_store.append(target_var)
            index_vector.append(index)
            
            # model.zero_grad()
            # set_requires_grad(model.decoder, True)
            # update D
            set_requires_grad(dis, True)
            optimizer2.zero_grad()
            dis_input = torch.cat([fea, fea_adv], dim=1)
            dis_pred = dis(dis_input.detach())
            dis_loss = criterion2(dis_pred, target_adv)
            gradient_scaler.scale(dis_loss).backward()
            gradient_scaler.step(optimizer2)
            set_requires_grad(dis, False)
            # optimizer1.zero_grad()
            fake_dis_pred = dis(dis_input)
            if args.adv:
                fake_dis_loss = criterion2(fake_dis_pred,torch.tensor(0.5).expand_as(fake_dis_pred).cuda())
            else:
                fake_dis_loss = criterion2(fake_dis_pred, 1-target_adv)

            uncertainty_loss = criterion1(pred_logit, target_var, triage_var, sigma)
            # ce_loss = criterion1(pred_logit, target_var)
            model_loss = uncertainty_loss + fake_dis_loss*args.lamb
        
        gradient_scaler.scale(model_loss).backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)
        gradient_scaler.step(optimizer1)
        scale = gradient_scaler.get_scale()
        
        gradient_scaler.update()        
        skip_lr_sched = (scale > gradient_scaler.get_scale())       

        uncertainty_losses.update(uncertainty_loss.item(), imgvar.size(0))
        fake_dis_losses.update(fake_dis_loss.item(), imgvar.size(0))

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'FakeLoss {fake_loss.val:.4f}({fake_loss.avg:.4f})\t'
                  'UncertaintyLoss {uncertainty_loss.val:.4f}({uncertainty_loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader),
                          batch_time=batch_time,
                          fake_loss=fake_dis_losses,
                          uncertainty_loss=uncertainty_losses))
    ## calculate auc on train set ##        
    # index_vector = torch.cat(index_vector, dim=0)  # [42624]
    # index_matrix = []
    # for i in range(int(max(index_vector)) + 1):
    #     index_matrix.append(index_vector == i)
    # index_matrix = torch.stack(index_matrix, dim=0).cuda().float()  # list to array  --->  [66,42624]
    # output_store_fc = torch.cat(output_store_fc, dim=0)  # list to array  --->  [42624, 2]
    # target_store = torch.cat(target_store, dim=0).float()  # [97000]
    # pred_matrix_fc = index_matrix.mm(output_store_fc)  # [52,97000] * [97000, 2] = [52,2]
    # pred_matrix_fc = pred_matrix_fc.div(index_matrix.sum(1).unsqueeze(dim=1))
    # assert torch.all(index_matrix.sum(1) != 0).item(), "index matrix error!"
    # target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(index_matrix.sum(1)).long()

    # prec_video, preds = model_util.accuracy(pred_matrix_fc.cpu(), target_vector.cpu())
    # result_dict = {}
    # result_name = []
    # for i, line in enumerate(train_dataset.get_name()):
    #     s1 = line.find('/')
    #     s2 = line[s1:].find(' ') + s1
    #     if not line[s1 + 1:s2] in result_dict:
    #         result_dict[line[s1 + 1:s2]] = 0
    #     result_dict[line[s1 + 1:s2]] += preds.numpy()[0][i]
    #     result_name.append(line[s1 + 1:s2])

   # print('sigma_vector: ', sigma_vector)

    writer.add_scalar('Uncertainty Loss/train', uncertainty_losses.avg, epoch)
    writer.add_scalar('DIS Loss/train', fake_dis_losses.avg, epoch)
    # wandb.log({"CE Loss/train": ce_losses.avg, "DIS Loss/train": fake_dis_losses.avg})
    
    return skip_lr_sched # preds, pred_matrix_fc.cpu()[:, 1], target_vector.cpu() 
    

def validate(val_loader, model, val_dataset):
    global record_
    batch_time = model_util.AverageMeter()
    VideoAcc = model_util.AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    output_store_fc = []
    target_store = []
    index_vector = []
    sigma_vector = []
    with torch.no_grad():
        for i, (input_first, spec_img, audio_first, target_first, triage_first, index, filename) in enumerate(val_loader):
            target = target_first.cuda()
            imgvar = input_first.cuda()  # input: [16,3, 128,128] target:[16]
            astvar = audio_first.cuda()
            # compute output
            ''' model & full_model'''
            # pred_score = model(imgvar, specvar, astvar, phrase='eval')
            pred_score, sigma = model(imgvar, astvar, phrase='eval')
            sigma_vector.append([sigma.cpu().numpy(), filename, target_first, triage_first]) 
            #TODO how to visualize?
            
            output_store_fc.append(pred_score)
            target_store.append(target)
            index_vector.append(index)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        index_vector = torch.cat(index_vector, dim=0)  # [42624]
        index_matrix = []
        for i in range(int(max(index_vector)) + 1):
            index_matrix.append(index_vector == i)
        index_matrix = torch.stack(index_matrix, dim=0).cuda().float()  # list to array  --->  [66,42624]
        output_store_fc = torch.cat(output_store_fc, dim=0)  # list to array  --->  [42624, 2]
        target_store = torch.cat(target_store, dim=0).float()  # [97000]
        pred_matrix_fc = index_matrix.mm(output_store_fc)  # [52,97000] * [97000, 2] = [52,2]
        pred_matrix_fc = pred_matrix_fc.div(index_matrix.sum(1).unsqueeze(dim=1))
        assert torch.all(index_matrix.sum(1) != 0).item(), "index matrix error!"
        target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(index_matrix.sum(1)).long()

        prec_video, preds = model_util.accuracy(pred_matrix_fc.cpu(), target_vector.cpu())
        result_dict = {}
        result_name = []
        for i, line in enumerate(val_dataset.get_name()):
            s1 = line.find('/')
            s2 = line[s1:].find(' ') + s1
            if not line[s1 + 1:s2] in result_dict:
                result_dict[line[s1 + 1:s2]] = 0
            result_dict[line[s1 + 1:s2]] += preds.numpy()[0][i]
            result_name.append(line[s1 + 1:s2])
        print(result_dict)
        VideoAcc.update(prec_video, int(max(index_vector)) + 1)
        print(' *Prec@VideoClassifier {VideoAcc.avg:.3f} '.format(VideoAcc=VideoAcc))
        return VideoAcc.avg, preds, pred_matrix_fc.cpu()[:, 1], target_vector.cpu(), result_name, sigma_vector


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()