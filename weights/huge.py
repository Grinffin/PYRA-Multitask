import timm
import torch

# a = timm.create_model("vit_huge_patch14_224_in21k", pretrained=True)
# torch.save(a, "/home/xiongyizhe/DualEfficient/dualefficient/weights/imagenet21k_ViT-H_14.pt")

a = torch.load("/home/xiongyizhe/DualEfficient/dualefficient/weights/imagenet21k_ViT-H_14.pt")
torch.save(a.state_dict(), "/home/xiongyizhe/DualEfficient/dualefficient/weights/imagenet21k_ViT-H_14.pth")