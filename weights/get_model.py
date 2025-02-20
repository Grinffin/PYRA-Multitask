import timm
import torch

# a = timm.create_model("vit_huge_patch14_224_in21k", pretrained=True)
# torch.save(a, "/nlp_group/xiongyizhe/research/dualefficient/dualefficient/weights/imagenet21k_ViT-H_14.pt")

# a = torch.load("/nlp_group/xiongyizhe/research/dualefficient/dualefficient/weights/imagenet21k_ViT-H_14.pt")
# torch.save(a.state_dict(), "/nlp_group/xiongyizhe/research/dualefficient/dualefficient/weights/imagenet21k_ViT-H_14.pth")

a = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
torch.save(a.state_dict(), "/home/xiongyizhe/DualEfficient/dualefficient/weights/DeiT-B_16.pth")

a = timm.create_model("deit_small_distilled_patch16_224", pretrained=True)
torch.save(a.state_dict(), "/home/xiongyizhe/DualEfficient/dualefficient/weights/DeiT-S_16.pth")

a = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
torch.save(a.state_dict(), "/home/xiongyizhe/DualEfficient/dualefficient/weights/DeiT-T_16.pth")