from .btcv import BTCV
from .amos import AMOS
from .prostate_mri import ProstateMRI
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset



def get_dataloader(args):
    transform_train = transforms.Compose([
         transforms.Resize((args.image_size,args.image_size)),
         transforms.ToTensor(),
     ])

    transform_train_seg = transforms.Compose([
         transforms.Resize((args.out_size,args.out_size)),
         transforms.ToTensor(),
     ])

    transform_test = transforms.Compose([
         transforms.Resize((args.image_size, args.image_size)),
         transforms.ToTensor(),
     ])

    transform_test_seg = transforms.Compose([
         transforms.Resize((args.out_size,args.out_size)),
         transforms.ToTensor(),
     ])
    
    if args.dataset == 'btcv':
        # BTCV dataset
        train_dataset = BTCV(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Training', prompt=args.prompt)
        test_dataset = BTCV(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Test', prompt=args.prompt)
        val_dataset = BTCV(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Validation', prompt=args.prompt)

    elif args.dataset == 'prostate_mri':
        # Prostate MRI dataset
        train_dataset = ProstateMRI(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Training', prompt=args.prompt)
        test_dataset = ProstateMRI(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Test', prompt=args.prompt)
        val_dataset = ProstateMRI(args, args.data_path, transform=transform_train, transform_msk=transform_train_seg, mode='Validation', prompt=args.prompt)
    
    else:
        raise ValueError("The dataset is not supported now!")

    # Use a subset for debugging if needed
    if args.debug:
        subset_size = args.subset_size if hasattr(args, 'subset_size') else 10
        train_dataset = Subset(train_dataset, list(range(min(len(train_dataset), subset_size))))
        val_dataset = Subset(val_dataset, list(range(min(len(val_dataset), subset_size))))
        test_dataset = Subset(test_dataset, list(range(min(len(test_dataset), subset_size))))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=1, pin_memory=True)  # Use a smaller batch size for testing/validation

    return train_loader, test_loader, val_loader