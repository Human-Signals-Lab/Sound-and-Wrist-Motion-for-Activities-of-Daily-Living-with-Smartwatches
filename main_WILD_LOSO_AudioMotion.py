import time
from ParticipantLab import ParticipantLab as parti
import numpy as np
# import tensorflow as tf
import sys
import pickle
import os
import copy

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
import matplotlib.pyplot as plt
import itertools
import csv

from sklearn import metrics
from enum import Enum
import librosa.display
import sys
from scipy import stats 
import datetime
from scipy.fftpack import dct
import _pickle as cPickle


from models import AttendDiscriminate_MotionAudio, Classifier
import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from utils_ import plotCNNStatistics
from collections import Counter
import random
torch.backends.cudnn.deterministic = True
random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# np.random.seed(1)


from sklearn import metrics

from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from models import create, StatisticsContainer
from utils.utils import paint, Logger, AverageMeter
from utils.utils_plot import plot_confusion
from utils.utils_pytorch import (
    get_info_params,
    get_info_layers,
    init_weights_orthogonal,
)
from utils.utils_mixup import mixup_data, MixUpLoss, mixup_data_AudioMotion, CrossEntropyLabelSmooth
from utils.utils_centerloss import compute_center_loss, get_center_delta

import warnings

warnings.filterwarnings("ignore")



def model_train(model, dataset, dataset_val, args):
    print(paint("[STEP 4] Running HAR training loop ..."))

    logger = SummaryWriter(log_dir=os.path.join(model.path_logs, "train"))
    logger_val = SummaryWriter(log_dir=os.path.join(model.path_logs, "val"))

    loader = DataLoader(dataset, args['batch_size'], True, pin_memory=True)
    loader_val = DataLoader(dataset_val, args['batch_size'], False, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()
    #criterion = CrossEntropyLabelSmooth(args['num_class'])
    params = filter(lambda p: p.requires_grad, model.parameters())

    if args['optimizer'] == "Adam":
        optimizer = optim.Adam(params, lr=args['lr'])
    elif args['optimizer'] == "RMSprop":
        optimizer = optim.RMSprop(params, lr=args['lr'])

    if args['lr_step'] > 0:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args['lr_step'], gamma=args['lr_decay']
        )

    # if args['init_weights'] == "orthogonal":
    #     print(paint("[-] Initializing weights (orthogonal)..."))
    #     model.apply(init_weights_orthogonal)

    print("[-] Loading checkpoint ...")
    if args['train_mode']:
        path_checkpoint = './models/train_LOPO_AttendDiscriminate_MotionAudio_CNN14/checkpoints/15/checkpoint_best.pth' #os.path.join(model.path_checkpoints, "checkpoint_best.pth")
    else:
        path_checkpoint = os.path.join(f"./weights/checkpoint.pth")

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])

    # model.classifier = Classifier(2048+128, 16).cuda()

    # print("Layer Name \t\t Parameter Size")
    # for name, param in model.named_parameters():
    #     if name not in ['classifier.fc.weight', 'classifier.fc.bias']:
    #         param.requires_grad = False

    #     print(name, "\t\t", param.size(), param.requires_grad)

    # import pdb; pdb.set_trace()

    metric_best = 0.0
    start_time = time.time()

    for epoch in range(args['epochs']):
        print("--" * 50)
        print("[-] Learning rate: ", optimizer.param_groups[0]["lr"])
        train_one_epoch(model, loader, criterion, optimizer, epoch, args)
        loss, acc, fm, rm, pm, fw = eval_one_epoch(
            model, loader, criterion, epoch, logger, args
        )
        loss_val, acc_val, fm_val, rm_val, pm_val, fw_val = eval_one_epoch(
            model, loader_val, criterion, epoch, logger_val, args
        )

        print(
            paint(
                f"[-] Epoch {epoch}/{args['epochs']}"
                f"\tTrain loss: {loss:.2f} \tacc: {acc:.2f}(%)\tfm: {fm:.2f}(%)\trm: {rm:.2f}(%)\tpm: {pm:.2f}(%)\tfw: {fw:.2f}(%)"
            )
        )

        print(
            paint(
                f"[-] Epoch {epoch}/{args['epochs']}"
                f"\tVal loss: {loss_val:.2f} \tacc: {acc_val:.2f}(%)\tfm: {fm_val:.2f}(%)\trm: {rm_val:.2f}(%)\tpm: {pm_val:.2f}(%)\tfw: {fw_val:.2f}(%)"
            )
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "random_rnd_state": random.getstate(),
            "numpy_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
        }

        metric = fm_val
        if metric >= metric_best:
            print(paint(f"[*] Saving checkpoint... ({metric_best}->{metric})", "blue"))
            metric_best = metric
            torch.save(
                checkpoint, os.path.join(model.path_checkpoints, "checkpoint_best.pth")
            )

        if epoch % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(model.path_checkpoints, f"checkpoint_{epoch}.pth"),
            )

        if args['lr_step'] > 0:
            scheduler.step()

        trainLoss = {'Trainloss': loss}
        args['statistics'].append(epoch, trainLoss, data_type='Trainloss')
        valLoss = {'Testloss': loss_val}
        args['statistics'].append(epoch, valLoss, data_type='Testloss')
        test_f1 = {'test_f1':fm}
        args['statistics'].append(epoch, test_f1, data_type='test_f1')

        args['statistics'].dump()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))

    print(paint(f"[STEP 4] Finished HAR training loop (h:m:s): {elapsed}"))
    print(paint("--" * 50, "blue"))



def train_one_epoch(model, loader, criterion, optimizer, epoch, args):

    losses = AverageMeter("Loss")
    model.train()

    for batch_idx, (data, data_a, target) in enumerate(loader):
        data = data.cuda()
        data_a = data_a.cuda()
        target = target.cuda()

        # centers = model.centers

        if args['mixup']:
            data, data_a, y_a_y_b_lam = mixup_data_AudioMotion(data, data_a, target, args['alpha'])

        z, logits = model(data, data_a)

        if args['mixup']:
            criterion = MixUpLoss(criterion)
            loss = criterion(logits, y_a_y_b_lam)
        else:
            loss = criterion(logits, target)

        # center_loss = compute_center_loss(z, centers, target)
        # loss = loss + args['beta'] * center_loss

        losses.update(loss.item(), data.shape[0])

        optimizer.zero_grad()
        loss.backward()
        if args['clip_grad'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad'])
        optimizer.step()

        # center_deltas = get_center_delta(z.data, centers, target, args['lr_cent'])
        # model.centers = centers - center_deltas

        if batch_idx % args['print_freq']== 0:
            print(f"[-] Batch {batch_idx}/{len(loader)}\t Loss: {str(losses)}")

        if args['mixup']:
            criterion = criterion.get_old()

def eval_one_epoch(model, loader, criterion, epoch, logger, args):

    losses = AverageMeter("Loss")
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, data_a, target) in enumerate(loader):
            data = data.cuda()
            data_a = data_a.cuda()
            target = target.cuda()

            z, logits = model(data,data_a)
            loss = criterion(logits, target.view(-1))
            losses.update(loss.item(), data.shape[0])

            probabilities = nn.Softmax(dim=1)(logits)
            _, predictions = torch.max(probabilities, 1)

            y_pred.append(predictions.cpu().numpy().reshape(-1))
            y_true.append(target.cpu().numpy().reshape(-1))

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)
    Y_True.extend(y_true)
    Y_Pred.extend(y_pred)
    # y_pred[y_pred == 1] = 0
    # y_pred[y_pred == 4] = 3
    # y_pred[y_pred== 5] = 3
    # y_pred[y_pred == 12] = 13

    acc = 100.0 * metrics.accuracy_score(y_true, y_pred)
    fm = 100.0 * metrics.f1_score(y_true, y_pred, average="macro")
    rm = 100.0* metrics.recall_score(y_true, y_pred, average="macro")
    pm = 100.0*metrics.precision_score(y_true, y_pred, average="macro")
    fw = 100.0 * metrics.f1_score(y_true, y_pred, average="weighted")

    if logger:
        logger.add_scalars("Loss", {"CrossEntropy": losses.avg}, epoch)
        logger.add_scalar("Acc", acc, epoch)
        logger.add_scalar("Fm", fm, epoch)
        logger.add_scalar("Rm", rm, epoch)
        logger.add_scalar("Pm", pm, epoch)
        logger.add_scalar("Fw", fw, epoch)

    if epoch % 50 == 0 or not args['train_mode']:
        plot_confusion(
            y_true,
            y_pred,
            os.path.join(model.path_visuals, f"cm/{args['participant']}"),
            epoch,
            class_map=args['class_map'],
        )

    return losses.avg, acc, fm, rm, pm, fw

def model_eval(model, dataset_test, args):
    print(paint("[STEP 5] Running HAR evaluation loop ..."))

    loader_test = DataLoader(dataset_test, args['batch_size'], False, pin_memory=True)

    criterion = nn.CrossEntropyLoss(reduction="mean").cuda()

    print("[-] Loading checkpoint ...")
    if args['train_mode']:
        path_checkpoint = os.path.join(model.path_checkpoints, "checkpoint_best.pth")
    else:
        path_checkpoint = os.path.join(f"./weights/checkpoint.pth")

    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion.load_state_dict(checkpoint["criterion_state_dict"])

    start_time = time.time()

    loss_test, acc_test, fm_test, rm_test, pm_test, fw_test = eval_one_epoch(
        model, loader_test, criterion, -1, logger=None, args=args
    )

    print(
        paint(
            f"[-] Test loss: {loss_test:.2f}"
            f"\tacc: {acc_test:.2f}(%)\tfm: {fm_test:.2f}(%)\trm: {rm_test:.2f}(%)\tpm: {pm_test:.2f}(%)\tfw: {fw_test:.2f}(%)"
        )
    )
    results.writerow([str(args['participant']), str(pm_test), str(rm_test), str(fm_test), str(acc_test), str(fw_test)])

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print(paint(f"[STEP 5] Finished HAR evaluation loop (h:m:s): {elapsed}"))


if __name__ == '__main__':


    P = 15
    win_size = 10
    hop = .5

    participants = []



    if os.path.exists('../Data/rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test.pkl'):
        with open('../Data/rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test.pkl', 'rb') as f:
            participants = pickle.load(f)
    else:

        start = time.time()
        for j in range (1, P+1):
            pname = str(j).zfill(2)
            p = parti(pname, '../Data',win_size, hop, normalized = False)
            p.readRawAudioMotionData()
            participants.append(p)
            print('participant',j,'data read...')
        end = time.time()
        print("time for feature extraction: " + str(end - start))

        with open('../Data/rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_Test.pkl', 'wb') as f:
            pickle.dump(participants, f)

    X_LabM = np.empty((0,np.shape(participants[0].rawMdataX_s1)[1], np.shape(participants[0].rawMdataX_s1)[-1]))
    X_LabA = np.empty((0,np.shape(participants[0].rawAdataX_s1)[-1]))
    y_Lab = np.zeros((0, 1))
    for x in participants[:-1]:
        print("Adding data for Participant: " + x.name)
        X_LabM = np.vstack((X_LabM, x.rawMdataX_s1[:]))
        X_LabM = np.vstack((X_LabM, x.rawMdataX_s2[:]))
        X_LabA = np.vstack((X_LabA, x.rawAdataX_s1[:]))
        X_LabA = np.vstack((X_LabA, x.rawAdataX_s2[:]))
        y_Lab = np.vstack((y_Lab, x.rawdataY_s1))
        y_Lab = np.vstack((y_Lab, x.rawdataY_s2))    

    WILD_participants = []


    if os.path.exists('../Data/WILD_JW_HH_KH_LV_SK_rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_2Sessions.pkl'):   #(old data with only 1 session, os.path.exists('../Data/WILD_JW_HH_KH_LV_SK_rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + 'NEW.pkl'):  
        with open('../Data/WILD_JW_HH_KH_LV_SK_rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_2Sessions.pkl', 'rb') as f:
            WILD_participants = pickle.load(f)
    else:
        start = time.time()
        files = ['JW','HH','KH','LV','SK']
        for pname in files:
            print('participant',pname,'data read...')
            p = parti(pname, '../Data',win_size, hop)
            p.readRawAudioMotionData()
            WILD_participants.append(p)
        end = time.time()
        print("time for feature extraction: " + str(end - start))

        with open('../Data/WILD_JW_HH_KH_LV_SK_rawAudioSegmentedData_window_' + str(win_size) + '_hop_' + str(hop) + '_2Sessions.pkl', 'wb') as f:
            pickle.dump(WILD_participants, f)


    model_name = 'AttendDiscriminate_MotionAudio_CNN14'
    experiment = f'WILD_LOSO_{model_name}_withMixUP_20Epochs'

    config_model = {
        "model": model_name,
        "input_dim": 6,
        "hidden_dim": 128,
        "filter_num": 64,
        "filter_size": 5,
        "enc_num_layers": 2,
        "enc_is_bidirectional": False,
        "dropout": .5,
        "dropout_rnn": .25,
        "dropout_cls": .5,
        "activation": "ReLU",
        "sa_div": 1,
        "num_class": 23,
        "train_mode": True,
        "experiment": experiment,
        "window_size": 1024,
        "hop_size": 320,
        "fmin": 50, 
        "fmax": 11000,
        "mel_bins": 64,
        "sample_rate": 22050
    }

    # results_path = './performance_results/{}/{}/'.format(experiment,model_name)
    # if not os.path.exists(results_path):
    #     os.makedirs(results_path)

    # file = open(results_path + 'performance_results.csv', "w")
    # results = csv.writer(file)
    # results.writerow(["Participant", "Precision", "Recall", "F-Score", "Accuracy", "F-weighted"])

    args = copy.deepcopy(config_model)

    # X_trainM = X_trainM[(y_train != 23)[:,0]]
    # X_trainA = X_trainA[(y_train != 23)[:,0]]

    # y_train = y_train[y_train != 23]

    # X_testM[X_testM == 1.] = 0.
    # X_testM[X_testM == 4.] = 3.
    # X_testM[X_testM == 5.] = 3.
    # X_testM[X_testM == 12.] = 13.


    sessionsM = []
    sessionsA = []
    sessionsY = []

    for u in WILD_participants:
        sessionsM.append(u.rawMdataX_s1)
        sessionsM.append(u.rawMdataX_s2)
        sessionsA.append(u.rawAdataX_s1)
        sessionsA.append(u.rawAdataX_s2)
        sessionsY.append(u.rawdataY_s1)
        sessionsY.append(u.rawdataY_s2)

    # import pdb; pdb.set_trace()
    Y_True, Y_Pred = [],[]
    for i in range(len(sessionsM)):
        print("participant : " + str(i))
        args['participant'] = str(i)
        config_model['participant'] = str(i)


        X_trainM = np.empty((0,np.shape(WILD_participants[0].rawMdataX_s1)[1], np.shape(WILD_participants[0].rawMdataX_s1)[-1]))
        X_trainA = np.empty((0,np.shape(WILD_participants[0].rawAdataX_s1)[-1]))
        y_train = np.zeros((0, 1))
        trainM = sessionsM[:i]+sessionsM[i+1:]
        trainA = sessionsA[:i]+sessionsA[i+1:]
        trainY = sessionsY[:i]+sessionsY[i+1:]

        for m,a,y in zip(trainM,trainA,trainY):
            X_trainM = np.vstack((X_trainM, m[:]))
            X_trainA = np.vstack((X_trainA, a[:]))
            y_train = np.vstack((y_train, y))

        # wild_classes = np.unique(y_train)
        # import pdb; pdb.set_trace()

        # X_trainM = np.vstack((X_trainM, u.rawMdataX_s1))
        # X_trainA= np.vstack((X_trainA, u.rawAdataX_s1))
        # y_train = np.vstack((y_train, u.rawdataY_s1))

        X_testM = copy.deepcopy(sessionsM[i])
        X_testA = copy.deepcopy(sessionsA[i])
        y_test = copy.deepcopy(sessionsY[i])

        X_trainM = X_trainM[(y_train != 23)[:,0]]
        X_trainA = X_trainA[(y_train != 23)[:,0]]

        y_train = y_train[y_train != 23]   
        
        X_testM = X_testM[(y_test != 23)[:,0]]
        X_testA = X_testA[(y_test != 23)[:,0]]

        y_test = y_test[y_test != 23]     

        classes = np.unique(y_test).astype(int)

        # mapping = {}
        labels = set(np.unique(y_test))
        labels.update(set(np.unique(y_train)))
        # labels = list(labels)
        # for i in range(len(labels)):
        #     mapping[labels[i]] = i

        # for x in range(len(y_test)):
        #     y_test[x] = mapping[y_test[x]]
        # for x in range(len(y_train)):
        #     y_train[x] = mapping[y_train[x]]    

        # print(mapping)
        # config_model['num_class'] = len(mapping)
        # args['num_class'] = len(mapping)
        # import pdb; pdb.set_trace()
        # 
        # y_train = tf.keras.utils.to_categorical(y_train, num_classes=23, dtype='int32')
        # y_test = tf.keras.utils.to_categorical(y_test, num_classes=23, dtype='int32')
        
        #import pdb; pdb.set_trace()
        torch.cuda.empty_cache()

        # [STEP 3] create HAR models
        # config_model['experiment'] = f'LOPO_{model_name}'

        if torch.cuda.is_available():
            model = create(model_name, config_model).cuda()
            torch.backends.cudnn.benchmark = True
            sys.stdout = Logger(
                os.path.join(model.path_logs, f"log_main_{experiment}.txt")
            )

        args['batch_size']= 32
        args['optimizer']= 'Adam'
        args['clip_grad']= 0
        args['lr']= 0.001
        args['lr_decay']= 0.9
        args['lr_step']= 10
        args['mixup']= True
        args['alpha']= 0.8
        args['lr_cent']= 0.001
        args['beta']= 0.003
        args['print_freq']= 40
        args['init_weights'] = None
        args['epochs'] = 20
        args['class_map'] = [chr(a+97).upper() for a in list(range(23))]

        # show args
        print("##" * 50)
        print(paint(f"Experiment: {model.experiment}", "blue"))
        print(
            paint(
                f"[-] Using {torch.cuda.device_count()} GPU: {torch.cuda.is_available()}"
            )
        )
        get_info_params(model)
        get_info_layers(model)
        print("##" * 50)

        statistics_path = './statistics/LOPO/{}/participant_{}/batch_size={}/statistics.pkl'.format(
                    model_name,u.name, args['batch_size'])
        if not os.path.exists(os.path.dirname(statistics_path)):
            os.makedirs(os.path.dirname(statistics_path))


        # Statistics
        statistics_container = StatisticsContainer(statistics_path)
        args['statistics'] = statistics_container


        all_classes = np.unique(y_Lab)
        counts = Counter(y_train)
        train_classes = np.unique(y_train)
        top_count = counts.most_common(1)[0][1]
        y_train = y_train[:,None]
        for c in all_classes:
            if c not in train_classes or (c in counts and counts[c] < top_count):
                tempM, tempA, tempY = X_LabM[y_Lab[:,0] == c], X_LabA[y_Lab[:,0] == c], y_Lab[y_Lab[:,0] == c]
                cc = counts[c] if c in counts else 0
                indx = random.sample(list(np.arange(len(tempM))),min(top_count-cc, len(tempM)))
                add_dataM, add_dataA, add_dataY = tempM[indx], tempA[indx], tempY[indx]
                X_trainM = np.vstack((X_trainM, add_dataM[:]))
                X_trainA = np.vstack((X_trainA, add_dataA[:]))
                y_train = np.vstack((y_train, add_dataY))
        import pdb; pdb.set_trace()

        y_train = y_train.flatten()
        X_trainM = X_trainM.astype('float32')
        X_trainA = X_trainA.astype('float32')
        y_train = y_train.astype('float32')

        x_testM_tensor = torch.from_numpy(np.array(X_testM)).float()
        x_testA_tensor = torch.from_numpy(np.array(X_testA)).float()
        y_test_tensor = torch.from_numpy(np.array(y_test)).long()

        # import pdb; pdb.set_trace()
        x_trainM_tensor = torch.from_numpy(np.array(X_trainM)).float()
        x_trainA_tensor = torch.from_numpy(np.array(X_trainA)).float()
        y_train_tensor = torch.from_numpy(np.array(y_train)).long()

        test_data = TensorDataset(x_testM_tensor, x_testA_tensor, y_test_tensor)
        train_data = TensorDataset(x_trainM_tensor, x_trainA_tensor, y_train_tensor)
        val_data = TensorDataset(x_testM_tensor, x_testA_tensor, y_test_tensor)
        # args['experiment'] = f'LOPO_{model_name}'

        if i < 1:
            model_eval(model, test_data, args)
            continue

        # [STEP 4] train HAR models
        model_train(model, train_data, val_data, args)

        # [STEP 5] evaluate HAR models
        if not args['train_mode']:
            args["experiment"] = "inference_LOPO"
            model = create(model_name, config_model).cuda()
        model_eval(model, test_data, args)

        #plotCNNStatistics(statistics_path, u)
        del model, test_data, train_data, val_data
    clss = set(np.unique(Y_True))
    clss.update(set(np.unique(Y_Pred)))
    plot_confusion(Y_True, Y_Pred, None, 0, normalize=True, cmap=plt.cm.Blues, class_map=[chr(a+97).upper() for a in clss])

    file.close()
    plt.show()







#print(np.shape(X_trainA), np.shape(X_testA), np.shape(y_train))


