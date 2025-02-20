import os
import re


# VTAB-1k datasets
exp_name_list = ["cifar100", "caltech101", "dtd", "oxford_flowers102", "oxford_pet", "svhn", "sun397", "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy", "clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"]
exp_dir = 'outputs/240920_ViT-S_prompt_lora_4_plain_lora_batch_64'

exp_acc_list = []
for exp_name in exp_name_list:
    acc = None
    for name in os.listdir(exp_dir):
        exp_file = os.path.join(exp_dir, name)
        if not os.path.isdir(exp_file) and exp_name in exp_file:
            found_cur = True
            with open(exp_file, "r") as f:
                log_data = f.read()
                # print(log_data)
                acc = re.findall(r"Max accuracy: ([0-9\.]*?)%", log_data)
                try:
                    acc = float(acc[-1])
                except:
                    acc = "null"
            exp_acc_list.append(acc)
            break
    if acc is None:
        exp_acc_list.append("null")

acc_str = ""
for acc in exp_acc_list:
    try:
        acc_str = acc_str + "%.2f "%(acc)
    except:
        acc_str = acc_str + acc + " "
print(acc_str)
with open(os.path.join(exp_dir, exp_dir.split("/")[-1].strip()+".txt"), 'w') as f:
    f.write(acc_str)
