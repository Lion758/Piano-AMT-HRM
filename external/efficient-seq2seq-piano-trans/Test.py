import torch

# Assume 'model' is an instance of your defined architecture
# Load the checkpoint file (replace 'checkpoint_path.pth' with your file path)
checkpoint = torch.load('checkpoint_path.pth')
model.load_state_dict(checkpoint['model_state_dict']) # adjust key if needed

# Calculate total parameters
total_params = sum(p.numel() for p in model.parameters())

# To count only trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params}")
print(f"Total Trainable Parameters: {trainable_params}")