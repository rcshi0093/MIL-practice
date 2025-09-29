from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             confusion_matrix, f1_score, recall_score,
                             precision_score)  # 移除specificity_score
import scipy.stats as stats

# 计算95%置信区间（基于bootstrap方法）
def calculate_auc_ci(y_true, y_prob, n_boot=1000):
    if len(np.unique(y_true)) < 2:
        return (np.nan, np.nan)
    
    auc_scores = []
    for _ in range(n_boot):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc_boot = roc_auc_score(y_true[idx], y_prob[idx, 1])
            auc_scores.append(auc_boot)
        except:
            continue
    
    if len(auc_scores) < 10:
        return (np.nan, np.nan)
    
    return (np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))

# 计算指标的均值和95%置信区间（跨折）
def calculate_metric_ci(scores):
    scores = np.array(scores)
    scores = scores[~np.isnan(scores)]
    if len(scores) <= 1:
        return (np.nan, np.nan, np.nan)
    
    mean = np.mean(scores)
    std = np.std(scores)
    se = std / np.sqrt(len(scores))
    h = se * stats.t.ppf((1 + 0.95) / 2, len(scores) - 1)
    return (mean, mean - h, mean + h)

# 计算二分类评估指标（手动计算特异度，不依赖specificity_score）
def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    
    # 准确率
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    
    # AUC
    if len(np.unique(y_true)) == 2 and len(y_true) > 1:
        metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1])
    else:
        metrics['AUC'] = np.nan
    
    # 混淆矩阵组件
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()  # 从混淆矩阵中提取TN, FP, FN, TP
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # 灵敏度（召回率）
    metrics['Sensitivity'] = recall_score(y_true, y_pred) if (tp + fn) > 0 else np.nan
    
    # 特异度（手动计算：TN/(TN+FP)）
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    # 阳性预测值（PPV）
    metrics['PPV'] = precision_score(y_true, y_pred) if (tp + fp) > 0 else np.nan
    
    # 阴性预测值（NPV）
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    
    # F1分数
    metrics['F1'] = f1_score(y_true, y_pred) if (tp + fp + fn) > 0 else np.nan
    
    return metrics

# 命令行参数设置
parser = argparse.ArgumentParser(description='CLAM Enhanced Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_ER', 'task_2_tumor_subtyping'])
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
args = parser.parse_args()

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径设置
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

# 验证路径有效性
assert os.path.isdir(args.models_dir), f"Models directory not found: {args.models_dir}"
assert os.path.isdir(args.splits_dir), f"Splits directory not found: {args.splits_dir}"

# 保存实验设置
settings = {
    'task': args.task,
    'split': args.split,
    'save_dir': args.save_dir, 
    'models_dir': args.models_dir,
    'model_type': args.model_type,
    'drop_out': args.drop_out,
    'model_size': args.model_size,
    'embed_dim': args.embed_dim,
    'folds_evaluated': list(folds) if 'folds' in locals() else []
}

with open(os.path.join(args.save_dir, f'eval_experiment_{args.save_exp_code}.txt'), 'w') as f:
    print(settings, file=f)
print("Experiment settings:")
print(settings)

# 加载数据集
if args.task == 'task_1_ER':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/task_1_ER.csv',
        data_dir=os.path.join(args.data_root_dir),
        shuffle=False, 
        print_info=True,
        label_dict={'nonER': 0, 'ER': 1},
        patient_strat=False,
        ignore=[]
    )

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(
        csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
        data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
        shuffle=False, 
        print_info=True,
        label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
        patient_strat=False,
        ignore=[]
    )
else:
    raise NotImplementedError(f"Task {args.task} is not implemented")

# 确定评估的fold范围
if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold + 1)
ckpt_paths = [os.path.join(args.models_dir, f's_{fold}_checkpoint.pt') for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

# 初始化结果存储结构
fold_metrics_list = []  # 存储每个fold每个子集的详细指标
summary_data = {
    'train': {'AUC': [], 'Accuracy': [], 'F1': [], 'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': []},
    'val': {'AUC': [], 'Accuracy': [], 'F1': [], 'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': []},
    'test': {'AUC': [], 'Accuracy': [], 'F1': [], 'Sensitivity': [], 'Specificity': [], 'PPV': [], 'NPV': []}
}

if __name__ == "__main__":
    # 遍历每个fold进行评估
    for ckpt_idx in range(len(ckpt_paths)):
        fold = folds[ckpt_idx]
        ckpt_path = ckpt_paths[ckpt_idx]
        print(f"\n{'='*50}")
        print(f"Evaluating Fold {fold}")
        print(f"Checkpoint path: {ckpt_path}")
        print(f"{'='*50}")
        
        # 验证检查点文件是否存在
        if not os.path.isfile(ckpt_path):
            print(f"Warning: Checkpoint file not found for fold {fold}, skipping...")
            continue
        
        # 加载当前fold的数据集拆分
        csv_path = os.path.join(args.splits_dir, f'splits_{fold}.csv')
        if not os.path.isfile(csv_path):
            print(f"Warning: Split file {csv_path} not found, skipping fold {fold}...")
            continue
        
        datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
        
        # 评估每个子集（train/val/test）
        for split_name in ['train', 'val', 'test']:
            print(f"\nEvaluating {split_name} set for fold {fold}...")
            split_dataset = datasets[datasets_id[split_name]]
            
            # 调用eval函数获取预测结果
            try:
                model, patient_results, test_error, auc, df = eval(split_dataset, args, ckpt_path)
            except Exception as e:
                print(f"Error evaluating {split_name} set for fold {fold}: {str(e)}")
                continue
            
            # 提取真实标签、预测标签和预测概率
            y_true = df['Y'].values
            y_pred = df['Y_hat'].values
            y_prob = df[[f'p_{i}' for i in range(args.n_classes)]].values
            
            # 计算评估指标
            metrics = compute_metrics(y_true, y_pred, y_prob)
            
            # 计算AUC的95%置信区间
            auc_ci = calculate_auc_ci(y_true, y_prob)
            
            # 保存当前fold和子集的详细指标
            fold_metrics = {
                'Fold': fold,
                'Subset': split_name,
                'AUC': metrics['AUC'],
                'AUC_95CI_lower': auc_ci[0],
                'AUC_95CI_upper': auc_ci[1],
                'Accuracy': metrics['Accuracy'],
                'Sensitivity': metrics['Sensitivity'],
                'Specificity': metrics['Specificity'],
                'PPV': metrics['PPV'],
                'NPV': metrics['NPV'],
                'F1': metrics['F1']
            }
            fold_metrics_list.append(fold_metrics)
            
            # 将指标添加到汇总数据中
            for metric in summary_data[split_name].keys():
                summary_data[split_name][metric].append(metrics[metric])
        
        # 保存当前fold的结果
        fold_metrics_df = pd.DataFrame(fold_metrics_list)
        fold_metrics_df.to_csv(os.path.join(args.save_dir, 'fold_metrics.csv'), index=False)
        print(f"Fold {fold} results saved to fold_metrics.csv")
    
    # 生成汇总统计结果
    summary_rows = []
    for split in ['train', 'val', 'test']:
        data = summary_data[split]
        # 计算每个指标的均值和95%置信区间
        auc_mean, auc_ci_low, auc_ci_high = calculate_metric_ci(data['AUC'])
        acc_mean, acc_ci_low, acc_ci_high = calculate_metric_ci(data['Accuracy'])
        f1_mean, f1_ci_low, f1_ci_high = calculate_metric_ci(data['F1'])
        sens_mean, sens_ci_low, sens_ci_high = calculate_metric_ci(data['Sensitivity'])
        spec_mean, spec_ci_low, spec_ci_high = calculate_metric_ci(data['Specificity'])
        ppv_mean, ppv_ci_low, ppv_ci_high = calculate_metric_ci(data['PPV'])
        npv_mean, npv_ci_low, npv_ci_high = calculate_metric_ci(data['NPV'])
        
        # 计算标准差
        auc_std = np.nanstd(data['AUC']) if len(data['AUC']) > 1 else np.nan
        acc_std = np.nanstd(data['Accuracy']) if len(data['Accuracy']) > 1 else np.nan
        f1_std = np.nanstd(data['F1']) if len(data['F1']) > 1 else np.nan
        sens_std = np.nanstd(data['Sensitivity']) if len(data['Sensitivity']) > 1 else np.nan
        spec_std = np.nanstd(data['Specificity']) if len(data['Specificity']) > 1 else np.nan
        ppv_std = np.nanstd(data['PPV']) if len(data['PPV']) > 1 else np.nan
        npv_std = np.nanstd(data['NPV']) if len(data['NPV']) > 1 else np.nan
        
        summary_rows.append({
            'Subset': split,
            'AUC_mean': auc_mean,
            'AUC_std': auc_std,
            'AUC_95CI': f"({auc_ci_low:.4f}, {auc_ci_high:.4f})",
            'Accuracy_mean': acc_mean,
            'Accuracy_std': acc_std,
            'Accuracy_95CI': f"({acc_ci_low:.4f}, {acc_ci_high:.4f})",
            'F1_mean': f1_mean,
            'F1_std': f1_std,
            'Sensitivity_mean': sens_mean,
            'Sensitivity_std': sens_std,
            'Specificity_mean': spec_mean,
            'Specificity_std': spec_std,
            'PPV_mean': ppv_mean,
            'PPV_std': ppv_std,
            'NPV_mean': npv_mean,
            'NPV_std': npv_std,
            'Folds_evaluated': len([x for x in data['AUC'] if not np.isnan(x)])
        })
    
    # 保存汇总结果
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.save_dir, 'summary.csv'), index=False)
    
    print("\n" + "="*50)
    print("Evaluation completed successfully!")
    print(f"Detailed fold metrics saved to: {os.path.join(args.save_dir, 'fold_metrics.csv')}")
    print(f"Summary statistics saved to: {os.path.join(args.save_dir, 'summary.csv')}")
    print("="*50)
    