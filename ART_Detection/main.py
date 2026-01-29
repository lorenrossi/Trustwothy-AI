# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


"""
Object Detection Robustness Tester using Adversarial Robustness Toolbox (ART)

Enhanced version with detailed error analysis:
1. Class misclassification (object detected but wrong class)
2. Complete miss (object not detected at all)
3. Localization error (object detected but box significantly displaced)

Installation:
pip install adversarial-robustness-toolbox
pip install torch torchvision matplotlib pillow numpy
"""

from tester_class import ARTObjectDetectionRobustnessTester as Robustness_tester

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = None

# Initialize tester (use_art_wrapper=False to avoid compatibility issues)
tester = Robustness_tester(
    device=device,
    model = model
    iou_threshold=0.5,
    use_art_wrapper=False  # Use native PyTorch implementation
)

# Example 1: Single image test
print("\n" + "=" * 70)
print("SINGLE IMAGE TEST")
print("=" * 70)

image_path = "https://images.cocodataset.org/val2017/000000039769.jpg"  # Cats on couch

results = tester.test_robustness(
    image_path,
    attack_type='fgsm',
    epsilon=0.03,
    visualize=True
)

# Example 2: Batch test with multiple images
print("\n" + "=" * 70)
print("BATCH TEST")
print("=" * 70)

image_urls = [
    "https://images.cocodataset.org/val2017/000000039769.jpg",  # Cats
    "https://images.cocodataset.org/val2017/000000037777.jpg",  # Traffic
    "https://images.cocodataset.org/val2017/000000252219.jpg",  # Kitchen
    "https://images.cocodataset.org/val2017/000000174482.jpg",  # Giraffe
    "https://images.cocodataset.org/val2017/000000087038.jpg",  # Baseball
]

batch_results = tester.test_robustness_batch(
    image_urls,
    attack_type='fgsm',
    epsilon=0.03,
    visualize=False  # Set True to see each image (slower)
)

# Access batch statistics
print("\nBatch testing complete!")
