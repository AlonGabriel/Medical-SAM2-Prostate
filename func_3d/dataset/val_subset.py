from torch.utils.data import DataLoader

# Assuming args and data_path are defined
val_dataset = prostate_mri(args, data_path, mode='Validation', prompt='click')
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)  # Use a small batch size for testing
