from main import *
import torch

class Args_Yaleb:
    alpha=0.5
    batch_size=190
    batch_size_te=1096
    beta=0.1
    beta_anneal=1
    clf_act='prelu'
    clf_hidden_units=75
    clf_layers=2
    clf_path='/home/quan-tran/disentanglement/FarconVAE/data/bestclf/bestclf_yaleb.pth'
    clf_seq='fad'
    clip_val=2.0
    connection=2
    cont_xs=1
    data_name='yaleb'
    data_path='/home/quan-tran/disentanglement/FarconVAE/data/yaleb/'
    dec_act='prelu'
    dec_seq='f'
    drop_p=0.3
    early_stop=0
    enc_act='prelu'
    enc_seq='fba'
    encoder='lr'
    end_fac=0.001
    env_eps=0.15
    env_flag='nn'
    epochs=2000
    eval_model='lr'
    fade_in=1
    gamma=0.5
    hidden_units=100
    kernel='t'
    last_epmod=0
    last_epmod_eval=1
    latent_dim=100
    lr=0.001
    max_lr=0.03
    model_name='ours'
    model_path='./model_yaleb'
    n_features=504
    n_seed=10
    neg_slop=0.1
    patience=200
    pred_act='leaky'
    pred_seq='fba'
    result_path='./result_yaleb'
    run_mode='e2e'
    s_dim=5
    save_name='default'
    scheduler='one'
    seed=730
    tr_ratio=1.0
    vis=0
    vis_path='./TSNE/'
    wd=0.0001
    y_dim=38

args = Args_Yaleb()
device = torch.device('cpu')

main(args, device)