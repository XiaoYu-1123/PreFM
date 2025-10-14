from __future__ import print_function
import os
import math
import time
import argparse
import numpy as np
import prettytable as pt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader_PreFM import PreFM_UnAV_dataset
from network_PreFM import PreFM_On_AVEL_Net
from utils.eval_metrics import segment_level, event_level_all_UnAV
import datetime


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def calculate_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    return total_norm

def lr_warm_up_cos_anneal(optimizer, cur_epoch, warm_up_epoch, max_epoch, lr_min, lr_max):
    if cur_epoch < warm_up_epoch:
        lr = cur_epoch / warm_up_epoch * lr_max
    else:
        lr = (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_epoch-warm_up_epoch)/(max_epoch-warm_up_epoch)*math.pi)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_evaluation_result_table(mode, F_scores_dict):
    f_scores_tb = pt.PrettyTable()
    fields = ["Dataset", "f_s", 'pf_map' ,"f_e_01" , 'f_e_03', 'f_e_05','f_e_07','f_e_09', 'f_e_avg']
    f_scores_tb.field_names = fields
    F_scores_list = [mode] + ['{:.2f}'.format(F_scores_dict[key]) for key in fields if key != 'Dataset']
    f_scores_tb.add_row(F_scores_list)

    return f_scores_tb

def gaussian_weights(num_steps, center, sigma):
    t = torch.arange(num_steps, dtype=torch.float32)
    weights = torch.exp(-0.5 * ((t - center) / sigma)**2)
    weights = weights * (num_steps / weights.sum())
    return weights  # shape: [num_steps]

def kd_cos_loss( stu, tea,time_weights):

    cos = F.cosine_similarity(stu, tea, dim=-1)
    lens = time_weights.shape[0]
    weighted_loss = cos * time_weights.view(1, lens)

    return 1-weighted_loss.mean()

def train(args, model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    train_loss = {'total': 0, 'loss_cls': 0, 'loss_kd_curr': 0,'loss_kd_future': 0, 'loss_all': 0}

    for batch_idx, batch_data in enumerate(train_loader):
        audio, visual  = batch_data['audio'].to(device), batch_data['visual'].to(device) # B 10 768
        curr_label = batch_data['curr_label'].float().to(device) # B 10 100
        future_label = batch_data['future_label'].float().to(device) # B 10 100
        curr_f_label, future_f_label  = batch_data['curr_f_label'].to(device), batch_data['future_f_label'].to(device) # B 10 512
        all_label = torch.cat([curr_label, future_label], dim=1)

        batch_size = visual.size(0)

        optimizer.zero_grad()
        
        all_frame_logits, curr, curr_f, future, future_f = model(audio, visual, curr_f_label, future_f_label)
        time_weights1 = gaussian_weights(num_steps = 10, center = 9.5, sigma = 5).to(device)
        time_weights2 = gaussian_weights(num_steps = 5, center = -0.5, sigma = 2.5).to(device)
        l_a = F.binary_cross_entropy_with_logits(all_frame_logits, all_label, reduction='none')
        loss_cls = l_a * torch.cat([time_weights1, time_weights2], dim = 0).view(1, 10 + args.future_length, 1)
        loss_cls = loss_cls.mean()
        loss_kd_curr = kd_cos_loss(curr, curr_f, time_weights1)
        loss_kd_future = kd_cos_loss(future, future_f, time_weights2)

        loss = loss_cls + loss_kd_curr + loss_kd_future

        train_loss['loss_cls'] += (loss_cls.item()*batch_size)
        train_loss['loss_kd_future'] += (loss_kd_future.item()*batch_size)
        train_loss['loss_kd_curr'] += (loss_kd_curr.item()*batch_size)
        train_loss['loss_all'] += (loss.item()*batch_size)
        train_loss['total'] += batch_size

        loss.backward()
        if args.grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        total_grad_norm = calculate_grad_norm(model)
        optimizer.step()

        # Log loss every 20 iterations
        if batch_idx % (len(train_loader) // 20 ) == 0:
            print(f"Epoch [{epoch}], Iter [{batch_idx}/{len(train_loader)}]: "
                  f"Loss cls: {loss_cls.item():.4f},  "
                  f"Loss kd_curr: {loss_kd_curr.item():.4f},  "
                  f"Loss kd_future: {loss_kd_future.item():.4f},  "
                  f"Total Loss: {loss.item():.4f}")

    num_train_data = train_loss['total']
    train_loss = {key: (float(value) / num_train_data) for key, value in train_loss.items() if 'loss' in key}

    return train_loss

from sklearn.metrics import average_precision_score
def compute_perframe_map(pred, gt):
    """
    Calculate the per-frame mAP for a single video.

    Parameters:
        pred: numpy array of shape [T, 100], representing the predicted scores 
            (values between 0 and 1 after applying sigmoid).
        gt:   numpy array of shape [T, 100], representing the ground truth 
            (values of 0 or 1).

    Returns:
        per-frame mAP: the mean of all frame-wise AP values.
    """
    T = pred.shape[0]
    ap_list = []
    for i in range(T):
        if np.sum(gt[i]) == 0:
            ap = 0.0
        else:
            ap = average_precision_score(gt[i], pred[i])
        ap_list.append(ap)
    return np.mean(ap_list)

def eval(args, model, data_loader, label_dir, criterion, device, mode):

    model.eval()

    pf_map = []
    F_seg_av = []
    F_event_av = [[] for _ in range(9)]


    val_loss = {'total': 0, 'loss_curr': 0, 'loss_kd_curr': 0, 'loss_all': 0}

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            video_names = batch_data['name']
            lengths = batch_data['length']
            video_name = video_names[0]
            length = lengths[0]
            audio, visual = batch_data['audio'].to(device), batch_data['visual'].to(device)
            curr_label = batch_data['curr_label'].float().to(device) # B,10,100
            curr_f_label = batch_data['curr_f_label'].to(device) # B 10 512
            batch_size = visual.size(0)



            all_frame_logits, curr, curr_f = model(audio, visual, curr_f_label)
            # ================= Calculate Validation Loss ===================
            if criterion != None:
                time_weights = gaussian_weights(num_steps = 10, center = 9.5, sigma = 5).to(device)
                loss_curr = F.binary_cross_entropy_with_logits(all_frame_logits[:,:10,:], curr_label)
                loss_kd_curr = kd_cos_loss(curr, curr_f, time_weights)

                loss = loss_curr + loss_kd_curr

                val_loss['loss_curr'] += (loss_curr.item()*batch_size)
                val_loss['loss_kd_curr'] += (loss_kd_curr.item()*batch_size)
                val_loss['loss_all'] += (loss.item()*batch_size)
            val_loss['total'] += batch_size
            # ================================================================

            
            curr_prob = torch.sigmoid(all_frame_logits[:length,9,:]) # B,10,100-->B,100-->T,100
            Pre_av = curr_prob.cpu().detach().numpy() # T,100
            GT_av = np.load(os.path.join(label_dir, mode, video_name + '.npy')) # T 100

            seg_map = compute_perframe_map(Pre_av, GT_av)
            pf_map.append(seg_map)

            GT_av_t = np.transpose(GT_av) # 100,T
            Pre_av_t = np.transpose(Pre_av) # 100,T
            Pre_av_t = (Pre_av_t >= 0.5).astype(np.int_) # 100, T

            f_av = segment_level(Pre_av_t, GT_av_t)
            F_seg_av.append(f_av)

            true_length = Pre_av_t.shape[1]
            event_scores = event_level_all_UnAV(Pre_av_t, GT_av_t, true_length, tiou_list=np.arange(0.1, 1.0, 0.1))
            for i, score in enumerate(event_scores):
                F_event_av[i].append(score)


    F_scores = {
        'f_s': 100 * np.mean(np.array(F_seg_av)),
        'pf_map': 100 * np.mean(np.array(pf_map)),
        **{f'f_e_{i+1:02d}': 100 * np.mean(np.array(F_event_av[i])) for i in range(9)},
        'f_e_avg': 100 * np.mean([np.mean(F_event_av[i]) for i in range(9)]),
    }

    num_train_data = val_loss['total']
    val_loss = {key: (float(value) / num_train_data) for key, value in val_loss.items() if 'loss' in key}

    return F_scores, val_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Official Implementation of PreFM')

    parser.add_argument("--pd_dir_train", type=str, default='./data_info/UnAV-100/train_pd.csv',
                        help="pd_dir_train")
    parser.add_argument("--pd_dir_val", type=str, default='./data_info/UnAV-100/val_pd.csv',
                        help="pd_dir_val")
    parser.add_argument("--pd_dir_test", type=str, default='./data_info/UnAV-100/test_pd.csv',
                        help="pd_dir_test")
    
    parser.add_argument("--audio_dir", type=str, default='../dataset/UnAV-100/data/CLAP')
    parser.add_argument("--visual_dir", type=str, default='../dataset/UnAV-100/data/CLIP')

    parser.add_argument("--label_train", type=str, default="../dataset/UnAV-100/label/label_clip_clap/train")
    parser.add_argument("--label_val", type=str, default="../dataset/UnAV-100/label/label_clip_clap/val")
    parser.add_argument("--label_test", type=str, default="../dataset/UnAV-100/label/label_clip_clap/test")
    parser.add_argument("--f_label_train", type=str, default="../dataset/UnAV-100/label/feature_label/train")
    parser.add_argument("--f_label_val", type=str, default="../dataset/UnAV-100/label/feature_label/val")
    parser.add_argument("--f_label_test", type=str, default="../dataset/UnAV-100/label/feature_label/test")
    parser.add_argument("--label_dir", type=str, default="../dataset/UnAV-100/label/label_clip_clap")
    

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'val', 'test'],
                        help="which mode to use")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='the value for gradient clipping (0 means no gradient clipping)')

    # optimizer hyper-parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)

    # scheduler hyper-parameters
    parser.add_argument('--scheduler', type=str, default='steplr', help='which scheduler to use')
    parser.add_argument('--stepsize', type=int, default=10, help='step size of learning scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma of learning scheduler')
    parser.add_argument('--warm_up_epoch', type=int, default=10, help='the number of epochs for warm up')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='the minimum lr for lr decay')

    # model hyper-parameters
    parser.add_argument("--model", type=str, default='PreFM_On_AVEL_Net', help="which model to use")
    parser.add_argument("--input_v_dim", type=int, default=768)
    parser.add_argument("--input_a_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--norm_where", type=str, default="post_norm", choices=['post_norm', 'pre_norm'])

    parser.add_argument("--model_name", type=str,
                        help="the name for the model")
    parser.add_argument("--model_save_dir", type=str, default='models/',
                        help="where to save the trained model")

    parser.add_argument("--future_length", type=int, default=5)    


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.model_name == None:
        args.model_name = args.model            
    print('args =', args)

    if args.mode == 'train':
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        assert not os.path.exists(os.path.join(args.model_save_dir, args.model_name, timestamp)), "{} already exists. Please specify another model_name.".format(args.model_name)

        os.makedirs(os.path.join(args.model_save_dir, args.model_name, timestamp), exist_ok=False)
        args_dict = args.__dict__
        with open(os.path.join(args.model_save_dir, args.model_name, timestamp, "arguments.txt"), 'w') as f:
            f.writelines('-------------------------start-------------------------\n')
            for key, value in args_dict.items():
                f.writelines(key + ': ' + str(value) + '\n')
            f.writelines('--------------------------end--------------------------\n')


    # Set random seed and device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PreFM_On_AVEL_Net(args).to(device)

    if args.mode == 'train':

        train_dataset = PreFM_UnAV_dataset(mode=args.mode, pd_dir=args.pd_dir_train, audio_dir=args.audio_dir, visual_dir=args.visual_dir,
                                    label_dir=args.label_train, f_label_dir=args.f_label_train, f_lens = args.future_length)
        val_dataset   = PreFM_UnAV_dataset(mode='val', pd_dir=args.pd_dir_val, audio_dir=args.audio_dir, visual_dir=args.visual_dir,
                                    label_dir=args.label_val, f_label_dir=args.f_label_val, f_lens = args.future_length)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=16, pin_memory=True)

        test_dataset = PreFM_UnAV_dataset(mode='test', pd_dir=args.pd_dir_test, audio_dir=args.audio_dir, visual_dir=args.visual_dir,
                                    label_dir=args.label_test, f_label_dir=args.f_label_test, f_lens = args.future_length)
        test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=16, pin_memory=True)

        # Create loss function(s)
        criterion = nn.BCELoss()

        # Create optimizer, scheduler
        if args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)

        if args.scheduler == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        elif args.scheduler == 'warm_up_cos_anneal':
            print('Using hand-made lr scheduler')
        else:
            print('Warning! Not using any lr scheduler!')


        best_F = {'f_e_avg': 0.0}
        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            train_dataset._init_dataset()
            if args.scheduler == 'warm_up_cos_anneal':
                lr_warm_up_cos_anneal(optimizer, epoch, args.warm_up_epoch, args.epochs, args.lr_min, args.lr)

            cur_lr = optimizer.param_groups[0]['lr']
            start_time = time.time()
            train_loss_dict = train(args, model, train_loader, optimizer, criterion, epoch, device)
            
            if args.scheduler != 'warm_up_cos_anneal':
                scheduler.step()

            if epoch%6==0:
                F_scores, val_loss_dict = eval(args, model, val_loader, args.label_dir, criterion, device, mode='val')
                elapse_time = time.time() - start_time
                
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, args.model_name, "checkpoint_epoch_{}.pt".format(epoch)))
                if F_scores['f_e_avg'] > best_F['f_e_avg']:
                    best_F = F_scores
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(args.model_save_dir, args.model_name, "checkpoint_best.pt"))

                print('Epoch[{}/{}](Time:{:.2f} sec)(lr:{:.6f}) Train Loss: {:.3f} Val Loss: {:.3f}  Val : {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
                        epoch, args.epochs, elapse_time, cur_lr, train_loss_dict['loss_all'], val_loss_dict['loss_all'], F_scores['f_s'], F_scores['pf_map'], F_scores['f_e_05'], F_scores['f_e_avg']))
                

        print('-'*30)
        print('Best F scores (at epoch {}):'.format(best_epoch))
        f_scores_tb = get_evaluation_result_table(mode="Val", F_scores_dict=best_F)
        print(f_scores_tb)

        model.load_state_dict(torch.load(os.path.join(args.model_save_dir, args.model_name, "checkpoint_best.pt")))

        # Evaluation
        print('begin testing...')
        F_scores_test, _ = eval(args, model, test_loader, args.label_dir, criterion=None, device=device, mode='test')
        f_scores_tb = get_evaluation_result_table(mode="Test", F_scores_dict=F_scores_test)
        print('Evaluation result:')
        print(f_scores_tb)

    elif args.mode == 'test':
        test_dataset = PreFM_UnAV_dataset(mode='test', pd_dir=args.pd_dir_test, audio_dir=args.audio_dir, visual_dir=args.visual_dir,
                                    label_dir=args.label_test, f_label_dir=args.f_label_test, f_lens = args.future_length)
        test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=16, pin_memory=True)
        model.load_state_dict(torch.load(os.path.join(args.model_save_dir, args.model_name, "checkpoint_best_ori.pt")))

        # Evaluation
        print('begin testing...')
        F_scores_test, _ = eval(args, model, test_loader, args.label_dir, criterion=None, device=device, mode='test')
        f_scores_tb = get_evaluation_result_table(mode="Test", F_scores_dict=F_scores_test)
        print('Evaluation result:')
        print(F_scores_test)
        print(f_scores_tb)

    else:
        print('Please specify args.mode!')
        

if __name__ == '__main__':
    main()
