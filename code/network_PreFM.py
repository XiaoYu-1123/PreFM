import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, hidden_dim):
        super(Encoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.final_norm_a = nn.LayerNorm(hidden_dim)
        self.final_norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, norm_where, src_a, src_v, mask=None, src_key_padding_mask=None):

        for i in range(self.num_layers):
            src_a = self.layers[i](norm_where, src_a, src_v, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask, with_ca=True)
            src_v = self.layers[i](norm_where, src_v, src_a, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask, with_ca=True)

        if norm_where == "pre_norm":
            src_a = self.final_norm_a(src_a)
            src_v = self.final_norm_v(src_v)

        return src_a, src_v


class HANLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, norm_where, src_q, src_v, src_mask=None, src_key_padding_mask=None, with_ca=True):
        """Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        if norm_where == "post_norm":
            if with_ca:
                src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
                src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
                src_q = self.norm1(src_q)
            else:
                src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout12(src2)
                src_q = self.norm1(src_q)

            src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
            src_q = src_q + self.dropout2(src2)
            src_q = self.norm2(src_q)

            return src_q.permute(1, 0, 2)
        
        elif norm_where == "pre_norm":
            src_q_pre_norm = self.norm1(src_q)

            if with_ca:
                src1 = self.cm_attn(src_q_pre_norm, src_v, src_v, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]
                src2 = self.self_attn(src_q_pre_norm, src_q_pre_norm, src_q_pre_norm, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            else:
                src2 = self.self_attn(src_q_pre_norm, src_q_pre_norm, src_q_pre_norm, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)[0]

                src_q = src_q + self.dropout12(src2)

            src_q_pre_norm = self.norm2(src_q)
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q_pre_norm))))
            src_q = src_q + self.dropout2(src2)

            return src_q.permute(1, 0, 2)
        
        else:
            raise ValueError('norm_where should be pre_norm or post_norm')


class Time_Modality(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(Time_Modality, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_t_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_m_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, norm_where, src_q, src_v1, src_v2, src_mask=None, src_key_padding_mask=None, with_ca=True):
        src_q = src_q.permute(1, 0, 2)
        src_v1 = src_v1.permute(1, 0, 2)
        src_v2 = src_v2.permute(1, 0, 2)

        src1 = self.cm_t_attn(src_q, src_v1, src_v1, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]
        src3 = self.cm_m_attn(src_q, src_v2, src_v2, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]

        src_q = src_q + self.dropout11(src1) + self.dropout12(src2) + self.dropout13(src3)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)

        return src_q.permute(1, 0, 2)


class Single_Layer_Q(nn.Module):
    def __init__(self, d_model, nhead, q_num, dim_feedforward=512, dropout=0.1):
        super(Single_Layer_Q, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # A simple transformer layer with a learnable query
        self.learnable_query = nn.Parameter(torch.randn(1, q_num, d_model))
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.q_num = q_num

    def forward(self, norm_where, src_v, src_mask=None, src_key_padding_mask=None, with_ca=True):
        
        batch_size = src_v.size(0)
        src_q = self.learnable_query.expand(batch_size, self.q_num,-1)

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        src1 = self.attn(src_q, src_v, src_v, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)[0]

        src_q = src_q + self.dropout1(src1)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)

        return src_q.permute(1, 0, 2)
    


class PreFM_On_AVEL_Net(nn.Module):

    def __init__(self, args):
        super(PreFM_On_AVEL_Net, self).__init__()
        self.f_lens = args.future_length

        self.fc_all = nn.Linear(args.hidden_dim*2, 100)
        self.fc_a =  nn.Linear(args.input_a_dim, args.hidden_dim)
        self.fc_v = nn.Linear(args.input_v_dim, args.hidden_dim)
        self.fc_feature = nn.Linear(1536, 512)

        self.hat_encoder = Encoder(HANLayer(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim),
                                   num_layers=args.num_layers,
                                   hidden_dim=args.hidden_dim)
        
        self.query_future_a = Single_Layer_Q(d_model=args.hidden_dim, nhead=args.nhead, q_num=self.f_lens, dim_feedforward=args.ff_dim)
        self.query_future_v = Single_Layer_Q(d_model=args.hidden_dim, nhead=args.nhead, q_num=self.f_lens, dim_feedforward=args.ff_dim)
        self.time_modality_a = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_v = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_a_f = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_v_f = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)

        self.norm_where = args.norm_where
        self.input_v_dim = args.input_v_dim     # 2048: ResNet152, 768: CLIP large
        self.input_a_dim = args.input_a_dim     # 128: VGGish, 512: CLAP
        self.hidden_dim = args.hidden_dim
        self.f_lens = args.future_length # 5

    def forward(self, audio, visual, curr_f_label, future_f_label=None):

        if audio.size(1) == 64:     # input data are feature maps
            x1 = audio.permute(0, 2, 1).contiguous().view(-1, self.input_a_dim, 2, 32)
            upsampled = F.interpolate(x1, size=(2, 1024), mode='bicubic')
            upsampled = self.fc_a(upsampled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).mean(dim=2)
            x1 = F.adaptive_avg_pool1d(upsampled, 10).view(-1, self.hidden_dim, 10)
            x1 = x1.permute(0, 2, 1)
        else:
            x1 = self.fc_a(audio)


        # 2d and 3d visual feature fusion
        if visual.size(1) == 80:        # input 2d features are from ResNet152
            vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
            vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        else:                           # input 2d features are from CLIP
            vid_s = self.fc_v(visual)

        x2 = vid_s

        # HAN
        x1, x2 = self.hat_encoder(self.norm_where, x1, x2)

        x1_f = self.query_future_a(self.norm_where, x1)
        x2_f = self.query_future_v(self.norm_where, x2)

        x1_f_aug = self.time_modality_a_f(self.norm_where, x1_f, x2_f, x1)
        x2_f_aug = self.time_modality_v_f(self.norm_where, x2_f, x1_f, x2)

        future = torch.cat([x1_f_aug, x2_f_aug], dim=-1)

        x1_aug = self.time_modality_a(self.norm_where, x1, x1_f_aug, x2)
        x2_aug = self.time_modality_v(self.norm_where, x2, x2_f_aug, x1)

        curr = torch.cat([x1_aug, x2_aug], dim=-1) 
        all = torch.cat([curr, future], dim=1)
        all_logits = self.fc_all(all)
        if self.training:
            curr_f = self.fc_feature(curr_f_label)
            future_f = self.fc_feature(future_f_label)
            return all_logits, curr, curr_f, future, future_f
        else:
            curr_f = self.fc_feature(curr_f_label)
            return all_logits, curr, curr_f
    


class PreFM_plus_On_AVEL_Net(nn.Module):

    def __init__(self, args):
        super(PreFM_plus_On_AVEL_Net, self).__init__()
        self.f_lens = args.future_length

        self.fc_all = nn.Linear(args.hidden_dim*2, 100)
        self.fc_a =  nn.Linear(args.input_a_dim, args.hidden_dim)
        self.fc_v = nn.Linear(args.input_v_dim, args.hidden_dim)
        self.fc_feature = nn.Linear(1536, 1024)

        self.hat_encoder = Encoder(HANLayer(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim),
                                   num_layers=args.num_layers,
                                   hidden_dim=args.hidden_dim)
        
        self.query_future_a = Single_Layer_Q(d_model=args.hidden_dim, nhead=args.nhead, q_num=self.f_lens, dim_feedforward=args.ff_dim)
        self.query_future_v = Single_Layer_Q(d_model=args.hidden_dim, nhead=args.nhead, q_num=self.f_lens, dim_feedforward=args.ff_dim)
        self.time_modality_a = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_v = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_a_f = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_v_f = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)

        self.norm_where = args.norm_where
        self.input_v_dim = args.input_v_dim     # 2048: ResNet152, 768: CLIP large
        self.input_a_dim = args.input_a_dim     # 128: VGGish, 512: CLAP
        self.hidden_dim = args.hidden_dim
        self.f_lens = args.future_length # 10

    def forward(self, audio, visual, curr_f_label, future_f_label=None):


        x1 = self.fc_a(audio)
        x2 = self.fc_v(visual)

        # HAN
        x1, x2 = self.hat_encoder(self.norm_where, x1, x2) # B 30 512

        x1_f = self.query_future_a(self.norm_where, x1) # B 10 512
        x2_f = self.query_future_v(self.norm_where, x2)

        x1_f_aug = self.time_modality_a_f(self.norm_where, x1_f, x2_f, x1) # B 10 512
        x2_f_aug = self.time_modality_v_f(self.norm_where, x2_f, x1_f, x2)

        future = torch.cat([x1_f_aug, x2_f_aug], dim=-1) # B 10 1024

        x1_aug = self.time_modality_a(self.norm_where, x1, x1_f_aug, x2) # B 30 512
        x2_aug = self.time_modality_v(self.norm_where, x2, x2_f_aug, x1)

        curr = torch.cat([x1_aug, x2_aug], dim=-1) # B 30 1024
        all = torch.cat([curr, future], dim=1) # B 40 1024 
        all_logits = self.fc_all(all)
        if self.training:
            curr_f = self.fc_feature(curr_f_label) # B 30 1024
            future_f = self.fc_feature(future_f_label) # B 10 1024
            return all_logits, curr, curr_f, future, future_f
        else:
            curr_f = self.fc_feature(curr_f_label)
            return all_logits, curr, curr_f



class PreFM_On_AVVP_Net(nn.Module):

    def __init__(self, args):
        super(PreFM_On_AVVP_Net, self).__init__()
        self.f_lens = args.future_length

        self.fc_a =  nn.Linear(args.input_a_dim, args.hidden_dim)
        self.fc_v = nn.Linear(args.input_v_dim, args.hidden_dim)
        self.fc_st = nn.Linear(512, args.hidden_dim)
        self.fc_fusion = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.fc_all = nn.Linear(args.hidden_dim, 25)
        self.fc_feature = nn.Linear(1536, args.hidden_dim)

        self.hat_encoder = Encoder(HANLayer(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim),
                                   num_layers=args.num_layers,
                                   hidden_dim=args.hidden_dim)
        
        self.query_future_a = Single_Layer_Q(d_model=args.hidden_dim, nhead=args.nhead, q_num=self.f_lens, dim_feedforward=args.ff_dim)
        self.query_future_v = Single_Layer_Q(d_model=args.hidden_dim, nhead=args.nhead, q_num=self.f_lens, dim_feedforward=args.ff_dim)
        self.time_modality_a = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_v = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_a_f = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)
        self.time_modality_v_f = Time_Modality(d_model=args.hidden_dim, nhead=args.nhead, dim_feedforward=args.ff_dim)

        self.norm_where = args.norm_where
        self.input_v_dim = args.input_v_dim     # 2048: ResNet152, 768: CLIP large
        self.input_a_dim = args.input_a_dim     # 128: VGGish, 512: CLAP
        self.hidden_dim = args.hidden_dim

    def forward(self, audio, visual, st, curr_f_label, future_f_label=None):

        if audio.size(1) == 64:     # input data are feature maps
            x1 = audio.permute(0, 2, 1).contiguous().view(-1, self.input_a_dim, 2, 32)
            upsampled = F.interpolate(x1, size=(2, 1024), mode='bicubic')
            upsampled = self.fc_a(upsampled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).mean(dim=2)
            x1 = F.adaptive_avg_pool1d(upsampled, 10).view(-1, self.hidden_dim, 10)
            x1 = x1.permute(0, 2, 1)
        else:
            x1 = self.fc_a(audio)


        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual)
        vid_st = self.fc_st(st)

        x2 = torch.cat((vid_s, vid_st), dim=-1)
        x2 = self.fc_fusion(x2)

        # HAN
        x1, x2 = self.hat_encoder(self.norm_where, x1, x2) # B 10 256


        x1_f = self.query_future_a(self.norm_where, x1)
        x2_f = self.query_future_v(self.norm_where, x2)

        x1_f_aug = self.time_modality_a_f(self.norm_where, x1_f, x2_f, x1) # B 5 256
        x2_f_aug = self.time_modality_v_f(self.norm_where, x2_f, x1_f, x2)

        futu_f_pred = torch.stack([x1_f_aug, x2_f_aug], dim=2) # B 5 2 256

        x1_aug = self.time_modality_a(self.norm_where, x1, x1_f_aug, x2) # B 10 256
        x2_aug = self.time_modality_v(self.norm_where, x2, x2_f_aug, x1)

        curr_f_pred = torch.stack([x1_aug, x2_aug], dim=2) # B 10 2 256

        all = torch.cat([curr_f_pred, futu_f_pred], dim=1) # B 15 2 256
        all_logits = self.fc_all(all) # B 15 2 25
    
        if self.training:
            curr_f = self.fc_feature(curr_f_label)
            futu_f = self.fc_feature(future_f_label)
            return all_logits, curr_f_pred, curr_f, futu_f_pred, futu_f
        else:
            curr_f = self.fc_feature(curr_f_label)
            return all_logits, curr_f_pred, curr_f