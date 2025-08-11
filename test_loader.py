# test_loader.py

from cub_loader import load_data, find_class_imbalance # adjust path if needed

# Load data
loader = load_data(['data/CUB_processed/train.pkl'], batch_size=4)
class_imbalance = find_class_imbalance('data/CUB_processed/train.pkl')

# Get one batch
batch = next(iter(loader))
x, c, y = batch

# Print shapes
print("Images:", x.shape)      # e.g. torch.Size([4, 3, 224, 224])
print("Concepts:", c.shape)    # e.g. torch.Size([4, 200])
print("Labels:", y.shape)      # e.g. torch.Size([4])

print("Class Imbalance:", class_imbalance)  # e.g. {0: 100, 1: 50, ...}\
