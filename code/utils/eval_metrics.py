import os
import csv
import numpy as np
import pandas as pd

import torch


def event_level_all_LLP(SO_av, GT_av, length, tiou_list):
    # 假定SO_av, GT_av为形状为 (N, sequence_length) 的数组，N=100
    N = 25
    # 预先提取每个样本的事件，避免重复计算
    event_p_av = [None for _ in range(N)]
    event_gt_av = [None for _ in range(N)]
    
    for n in range(N):
        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            event_p_av[n] = extract_event(seq_pred, n, length)
        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            event_gt_av[n] = extract_event(seq_gt, n, length)
    
    # 针对每个tiou阈值计算对应的f_av
    f_av_list = []
    for threshold in tiou_list:
        TP_av = np.zeros(N)
        FP_av = np.zeros(N)
        FN_av = np.zeros(N)
        TN_av = np.zeros(N)
        
        # 遍历每个样本，计算该阈值下的各项指标
        for n in range(N):
            tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n], threshold)
            TP_av[n] += tp
            FP_av[n] += fp
            FN_av[n] += fn
            # 计算 TN (用于准确率等计算，原公式：(TP + TN) / (TP+TN+FP+FN))
            TN_av[n] = length - (TP_av[n] + FP_av[n] + FN_av[n])
        
        # 计算F-score
        F_av = []
        for ii in range(N):
            # 避免分母为0
            if (TP_av[ii] + FP_av[ii]) != 0 or (TP_av[ii] + FN_av[ii]) != 0:
                F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av[ii] + FP_av[ii])))
        
        if len(F_av) == 0:
            f_av = 1.0  # 如果所有样本均为真负，定义f_av为1.0
        else:
            f_av = sum(F_av) / len(F_av)
        f_av_list.append(f_av)
    
    return f_av_list

def event_level_all_UnAV(SO_av, GT_av, length, tiou_list):
    # 假定SO_av, GT_av为形状为 (N, sequence_length) 的数组，N=100
    N = 100
    # 预先提取每个样本的事件，避免重复计算
    event_p_av = [None for _ in range(N)]
    event_gt_av = [None for _ in range(N)]
    
    for n in range(N):
        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            event_p_av[n] = extract_event(seq_pred, n, length)
        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            event_gt_av[n] = extract_event(seq_gt, n, length)
    
    # 针对每个tiou阈值计算对应的f_av
    f_av_list = []
    for threshold in tiou_list:
        TP_av = np.zeros(N)
        FP_av = np.zeros(N)
        FN_av = np.zeros(N)
        TN_av = np.zeros(N)
        
        # 遍历每个样本，计算该阈值下的各项指标
        for n in range(N):
            tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n], threshold)
            TP_av[n] += tp
            FP_av[n] += fp
            FN_av[n] += fn
            # 计算 TN (用于准确率等计算，原公式：(TP + TN) / (TP+TN+FP+FN))
            TN_av[n] = length - (TP_av[n] + FP_av[n] + FN_av[n])
        
        # 计算F-score
        F_av = []
        for ii in range(N):
            # 避免分母为0
            if (TP_av[ii] + FP_av[ii]) != 0 or (TP_av[ii] + FN_av[ii]) != 0:
                F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av[ii] + FP_av[ii])))
        
        if len(F_av) == 0:
            f_av = 1.0  # 如果所有样本均为真负，定义f_av为1.0
        else:
            f_av = sum(F_av) / len(F_av)
        f_av_list.append(f_av)
    
    return f_av_list



def segment_level(SO_av, GT_av):

    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    TN_av = np.sum((1 - SO_av) * (1 - GT_av), axis=1)

    n = len(FP_av)
    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))


    
    if len(F_av) == 0:
        f_av = 1.0  # all true negatives
    else:
        f_av = (sum(F_av) / len(F_av))  # average across classes

    # 计算每个类别的accuracy: (TP + TN) / (TP + TN + FP + FN)
    total = TP_av + TN_av + FP_av + FN_av
    # 避免除以0的情况，如果总数为0则定义accuracy为1.0
    acc_each = np.where(total == 0, 1.0, (TP_av + TN_av) / total)
    acc_av = np.mean(acc_each)

    return f_av


def to_vec(start, end, length):
    x = np.zeros(length)
    for i in range(start, end):
        x[i] = 1
    return x


def extract_event(seq, n, length):
    x = []
    i = 0
    while i < length:
        if seq[i] == 1:
            start = i
            if i + 1 == length:
                i = i + 1
                end = i
                x.append(to_vec(start, end, length))
                break

            for j in range(i + 1, length):
                if seq[j] != 1:
                    i = j + 1
                    end = j
                    x.append(to_vec(start, end, length))
                    break
                else:
                    i = j + 1
                    if i == length:
                        end = i
                        x.append(to_vec(start, end, length))
                        break
        else:
            i += 1
    return x


def event_wise_metric(event_p, event_gt, tiou):
    TP = 0
    FP = 0
    FN = 0

    if event_p is not None:
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= tiou * np.sum(x1 + x2 - x1 * x2):  # IoU, threshold=0.5
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= tiou * np.sum(x1 + x2 - x1 * x2):  # 0.5
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
    return TP, FP, FN
