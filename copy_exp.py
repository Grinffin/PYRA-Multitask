import os
import shutil


# VTAB-1k datasets
exp_name_list = ["cifar100", "caltech101", "dtd", "oxford_flowers102", "oxford_pet", "svhn", "sun397", "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy", "clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"]
exp_dir = 'outputs/240910_ViT-B_prompt_lora_8_compress_high_PYRA_RUN_'
best_exp_dir = 'outputs/ViT-B_PYRA_logs_weights_high_compression_rate'
best_str = "3 10 10 1 7 9 1 10 1 8 9 4 9 8 3 2 6 5 8" 

if not os.path.exists(best_exp_dir):
    os.makedirs(best_exp_dir)

best_run = best_str.strip().split(" ")
assert len(best_run) == len(exp_name_list)
for exp_name in exp_name_list:
    now_exp_dir = exp_dir + best_run[exp_name_list.index(exp_name)]
    try:
        for name in os.listdir(now_exp_dir):
            exp_file = os.path.join(now_exp_dir, name)
            if exp_name in exp_file:
                # copy to best_exp_dir
                if os.path.isdir(exp_file):
                    dest_dir = os.path.join(best_exp_dir, name)
                    shutil.copytree(exp_file, dest_dir)
                else:
                    shutil.copy(exp_file, best_exp_dir)
    except:
        print("Error: %s is null"%exp_name)

