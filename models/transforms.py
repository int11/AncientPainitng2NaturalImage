import torch
from torchvision import transforms


def trans_vgg(content_image):
    """
    Prepares content image for neural networks.
    
    Args:
        content_image: Input tensor in RGB format
        mean_values: List of mean values for BGR channels. 
                    Default is ImageNet BGR mean [103.939, 116.779, 123.680]
    
    Returns:
        Tensor with channels swapped to BGR and mean subtracted
    """
    # Step 1: Denormalize from [-1,1] to [0,1]
    step1 = transforms.Lambda(lambda x: (x + 1.0) / 2.0)(content_image)
    
    # Step 2: Apply ImageNet normalization
    step2 = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),  # ImageNet RGB mean
        std=(0.229, 0.224, 0.225)   # ImageNet RGB std
    )(step1)
    
    return step2


def trans_dexinet(content_image):
    """
    Prepares content image for DexiNed network.
    
    Args:
        content_image: Input tensor in RGB format
    
    Returns:
        Tensor with channels swapped to BGR and mean subtracted
    """
    # Step 1: Denormalize from [-1,1] to [0,255]
    step1 = transforms.Lambda(lambda x: ((x + 1.0) / 2.0) * 255.0)(content_image)
    
    # Step 2: Subtract ImageNet mean pixel-wise
    step2 = transforms.Lambda(
        lambda x: x - torch.tensor([123.68, 116.779, 103.939], device=x.device).view(1, 3, 1, 1)
    )(step1)
    
    # Step 3: Convert RGB to BGR
    step3 = transforms.Lambda(lambda x: x[:, [2, 1, 0], :, :])(step2)

    return step3
