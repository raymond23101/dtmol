import os
import sys
import argparse

args = {
    'world_size': 2,
    'batch_size': 8,
    'model_name': "bindingpose",
    'data_f': "/home/haotiant/data/protein_ligand_binding_pose_prediction/",
    'train':{
        'learning_rate':1e-4,
        'start_lr_factor': 1e-2,
        'lr_scheduler': 'LinearLR',
        'lr_warmup': 5,
        'grad_norm': None,
        'epoches': 500,
        'report_every': 100,
        'valid_first_n': 10,
        'eval_every_n_epoches': 5,
        'fine_tune_pretrain': False,
        'norm_weighted': False, # if the perturbation loss is weighted by normalization factor
        'use_wandb': True,
        'retrain': None,
        'warmup': None,
        'perturbation_loss': True, #If include perturbation noise
        'trrot_loss':True, #If include trrot noise
        'mode': 'train', #Can be 'train', 'debug', 'test' mode
        'max_reverse_diffusion_time': 20, #The maximum diffusion time for reverse diffusion
        'stochastic_reverse_sampling': False, #If use stochastic sampling when doing evaluation
        'record_intermediate': None, #If record the intermediate results
    },
    'model':
    {
        'dropout':0.1,
        'rotation_weight': 1.0, #Currently have no effect
        'translation_weight': 1.0, #Currently have no effect
        'perturbation_weight': 1.0, #Currently have no effect
        'independent_se3_attention': True,
        'update_distance_matrix': False,
        'use_cross_product_update': False,
        },
    'dataset':
    {
        "seed": 0,
        "max_diffusion_time":5000,
        "max_seq_len": 768,
        "max_pocket_atoms": 256,
        'tr_sigma_min': 0.1,
        'tr_sigma_max': 0.9999,
        'tr_sde': 'VE',
        'rot_sigma_min': 0.1,
        'rot_sigma_max': 1.65,
        'rot_sde': 'VE',
        'pert_mole_sigma_min': 0.1,
        'pert_mole_sigma_max': 1,
        'pert_mole_sde': 'VE',
        'pert_prot_sigma_min': 0.1,
        'pert_prot_sigma_max': 1,
        'pert_prot_sde': 'VE',
        'prot_pert': True,
        'mole_pert': True,
        'trrot': True,
    }
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--data_f",type=str,default = None)
    parser.add_argument("--world_size",type=int,default=None)
    parser.add_argument("--batch_size",type=int,default=None)
    parser.add_argument("--model_name",type=str,default=None)

    ### Training arguments
    parser.add_argument("--fine_tune_pretrain",action='store_true',dest = 'fine_tune_pretrain')
    parser.add_argument("--warmup",type = int,default=None, 
                        help="This setting will override the fine_tune_pretrain setting, will fine tune the encoder after warmup epochs")
    parser.add_argument("--no_wandb",action='store_false',dest='use_wandb')
    parser.add_argument("--retrain",type = str,default=None)
    parser.add_argument("--learning_rate",type=float,default=None)
    parser.add_argument("--start_lr_factor",type=float,default=None)
    parser.add_argument("--grad_norm",type=float,default=None)
    parser.add_argument("--lr_scheduler",type=str,default=None,
                        help = "Choose from ['LinearLR','CosineAnnealingLR','CosineAnnealingWarmRestarts']")
    parser.add_argument("--lr_warmup",type=int,default=None)
    parser.add_argument("--no_perturbation_loss",action = 'store_false',dest='perturbation_loss', 
                        help = "This will override the prot_pert and mole_pert setting in the dataset")
    parser.add_argument("--no_trrot_loss", action = 'store_false', dest = 'trrot_loss',
                        help = "This will override the trrot setting in the dataset")
    parser.add_argument("--epoches",type=int,default=None)
    parser.add_argument("--report_every",type=int,default=None)
    parser.add_argument("--valid_first_n",type=int,default=None)
    parser.add_argument("--eval_every_n_epoches",type=int,default=None)
    parser.add_argument("--mode",type=str,default=None,help = "Can be 'train', 'debug', 'test' mode")
    parser.add_argument("--max_reverse_diffusion_time",type=int,default=None,
                        help = "The maximum diffusion time for reverse diffusion, default is 20.")
    parser.add_argument("--stochastic_reverse_sampling",action='store_true',dest='stochastic_reverse_sampling',
                        help = "If use stochastic sampling when doing evaluation")
    parser.add_argument("--record_intermediate",type = str,default=None,
                        help = "The folder to record intermediate results")

    ### Model settings
    parser.add_argument("--dropout",type=float,default=0.1)
    parser.add_argument("--rotation_weight",type=float,default=1.0)
    parser.add_argument("--translation_weight",type=float,default=1.0)
    parser.add_argument("--perturbation_weight",type=float,default=1.0)
    parser.add_argument("--share_attention",action='store_false',dest='independent_se3_attention',
                        help = "Use the same attention for the coordinates and the features")
    parser.add_argument("--update_distance_matrix",action='store_true',
                        help = "Update the distance matrix in the diffusion process")
    parser.add_argument("--cross_product_update",action='store_true',dest='use_cross_product_update',
                        help = "Enable the cross product update for the coordinates in the diffusion process")

    ### Diffusion settings
    parser.add_argument("--tr_sigma_min",type = float,default = None)
    parser.add_argument("--tr_sigma_max",type = float,default = None)
    parser.add_argument("--tr_sde", type = str, default = None)
    parser.add_argument("--rot_sigma_min",type = float, default = None)
    parser.add_argument("--rot_sigma_max",type = float, default = None)
    parser.add_argument("--rot_sde", type = str, default = None)
    parser.add_argument("--pert_mole_sigma_min",type = float, default = None)
    parser.add_argument("--pert_mole_sigma_max",type = float, default = None)
    parser.add_argument("--pert_mole_sde", type = str, default = "VE")
    parser.add_argument("--pert_prot_sigma_min",type = float, default = None)
    parser.add_argument("--pert_prot_sigma_max",type = float, default = None)
    parser.add_argument("--pert_prot_sde", type = str, default = "VE")
    parser.add_argument("--no_prot_pert", action = "store_false", dest = "prot_pert")
    parser.add_argument("--no_mole_pert", action = "store_false", dest = "mole_pert")
    parser.add_argument("--no_trrot", action = "store_false", dest = "trrot")
    cmd_args = vars(parser.parse_args(sys.argv[1:]))
    if not cmd_args['perturbation_loss']:
        cmd_args['prot_pert'] = False
        cmd_args['mole_pert'] = False
    if not cmd_args['trrot_loss']:
        cmd_args['trrot'] = False
    assert cmd_args['lr_scheduler'] in [None,'LinearLR','CosineAnnealingLR','CosineAnnealingWarmRestarts'], "lr_scheduler should be one of ['LinearLR','CosineAnnealingLR','CosineAnnealingWarmRestarts']"
    update_args(args,cmd_args)
    return args

def update_arg(args, key,value):
    """Update argument for any same name argument in the disctionry and sub-dictionary recursively.
    """
    if value is None:
        return True
    if key in args:
        args[key] = value
        return True
    else:
        for k,v in args.items():
            if isinstance(v,dict):
                state = update_arg(v,key,value)
                if state:
                    return True
    return False

def update_args(args, cmd_args):
    #update the args with the parsed args if parser is not None 
    for key in cmd_args:
        result = update_arg(args,key,cmd_args[key])
        if not result:
            print('Warning: key {key} is not found in the args, will create a new key in the base-level of args.')
            args[key] = cmd_args[key]


def print_args(args,level = 0):
    prefix = "\t"*level
    for key in args:
        if isinstance(args[key],dict):
            print(f"{prefix}{key}:")
            print_args(args[key],level+1)
        else:
            print(f"{prefix}{key}: {args[key]}")

if __name__ == "__main__":
    old_args = args
    test_args = {"retrain": False,
                 "dropout": 1.2,
                 "new_args":1.5}
    update_args(old_args,test_args)
    print(old_args)
