# test_loader.py

from dataset import load_data  # adjust path if needed

# Load data
loader = load_data(['data/CUB_processed/train.pkl'], batch_size=4)

# Get one batch
batch = next(iter(loader))
x, c, y = batch

# Print shapes
print("Images:", x.shape)      # e.g. torch.Size([4, 3, 224, 224])
print("Concepts:", c.shape)    # e.g. torch.Size([4, 200])
print("Labels:", y.shape)      # e.g. torch.Size([4])