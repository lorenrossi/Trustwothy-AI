

import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict
import urllib.request
from io import BytesIO
import ssl

# ART imports
from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.attacks.evasion import BasicIterativeMethod, CarliniL2Method


class ARTObjectDetectionRobustnessTester:
    """
    Tests robustness of object detection models using ART library.
    Provides detailed analysis of different types of adversarial failures.
    """

    def __init__(self, model=None, device='cpu', iou_threshold=0.5, use_art_wrapper=False):
        """
        Initialize the tester with a model.

        Args:
            model: PyTorch object detection model (default: Faster R-CNN)
            device: 'cpu' or 'cuda'
            iou_threshold: IoU threshold for matching detections (default: 0.5)
            use_art_wrapper: Whether to use ART wrapper (can have compatibility issues)
        """
        self.device = device
        self.iou_threshold = iou_threshold
        self.use_art_wrapper = use_art_wrapper

        # Load default model if none provided
        if model is None:
            print("Loading Faster R-CNN model...")
            self.pytorch_model = fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            self.pytorch_model = model

        self.pytorch_model.to(self.device)
        self.pytorch_model.eval()

        # Try to wrap model with ART (optional)
        if self.use_art_wrapper:
            print("Wrapping model with ART...")
            self.art_model = self._wrap_model_with_art()
        else:
            print("Using direct PyTorch implementation (bypassing ART wrapper)")
            self.art_model = None

        # COCO class names
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def _wrap_model_with_art(self):
        """Wrap PyTorch model with ART's PyTorchFasterRCNN wrapper."""
        try:
            art_model = PyTorchFasterRCNN(
                model=self.pytorch_model,
                clip_values=(0, 1),
                attack_losses=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
                device_type=self.device
            )
            return art_model
        except Exception as e:
            print(f"Warning: Could not wrap with ART PyTorchFasterRCNN: {e}")
            print("Falling back to generic wrapper...")
            return None

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Image.Image]:
        """
        Load and preprocess an image for ART.
        Supports both local file paths and URLs.

        Args:
            image_path: Local file path or URL (http:// or https://)
        """
        if image_path.startswith('http://') or image_path.startswith('https://'):
            print(f"Downloading image from URL: {image_path}")
            try:
                # Create SSL context that doesn't verify certificates
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                with urllib.request.urlopen(image_path, context=ssl_context) as response:
                    img_data = response.read()
                img = Image.open(BytesIO(img_data)).convert('RGB')
                print("✓ Image downloaded successfully")
            except Exception as e:
                raise ValueError(f"Failed to download image from URL: {e}")
        else:
            img = Image.open(image_path).convert('RGB')

        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array, img

    def detect_objects(self, img_array: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Run object detection on an image.

        Args:
            img_array: Image as numpy array [H, W, C] in [0,1] range
            threshold: Confidence threshold
        """
        # Convert to tensor for PyTorch model
        # Ensure correct shape: [C, H, W]
        if img_array.ndim == 3:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Expected 3D array [H,W,C], got shape {img_array.shape}")

        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.pytorch_model(img_tensor)[0]

        # Filter by confidence threshold
        keep = predictions['scores'] > threshold

        return {
            'boxes': predictions['boxes'][keep].cpu().numpy(),
            'labels': predictions['labels'][keep].cpu().numpy(),
            'scores': predictions['scores'][keep].cpu().numpy()
        }

    def create_attack(self, attack_type: str, **kwargs):
        """
        Create an attack object (ART or native implementation).

        Args:
            attack_type: One of 'fgsm', 'pgd', 'bim'
            **kwargs: Attack-specific parameters

        Returns:
            Attack configuration dict for native implementation
        """
        if self.use_art_wrapper and self.art_model is not None:
            # Use ART implementation
            attack_map = {
                'fgsm': FastGradientMethod,
                'pgd': ProjectedGradientDescent,
                'bim': BasicIterativeMethod,
            }

            attack_type_lower = attack_type.lower()
            if attack_type_lower not in attack_map:
                raise ValueError(f"Unknown attack: {attack_type}. Choose from {list(attack_map.keys())}")

            if attack_type_lower == 'fgsm':
                params = {'eps': kwargs.get('epsilon', 0.03), 'batch_size': 1}
            elif attack_type_lower == 'pgd':
                params = {
                    'eps': kwargs.get('epsilon', 0.03),
                    'eps_step': kwargs.get('eps_step', 0.01),
                    'max_iter': kwargs.get('max_iter', 10),
                    'batch_size': 1
                }
            elif attack_type_lower == 'bim':
                params = {
                    'eps': kwargs.get('epsilon', 0.03),
                    'eps_step': kwargs.get('eps_step', 0.01),
                    'max_iter': kwargs.get('max_iter', 10),
                    'batch_size': 1
                }

            print(f"Creating {attack_type.upper()} attack with ART: {params}")
            attack = attack_map[attack_type_lower](estimator=self.art_model, **params)
            return attack
        else:
            # Use native PyTorch implementation
            attack_config = {
                'type': attack_type.lower(),
                'epsilon': kwargs.get('epsilon', 0.03),
                'eps_step': kwargs.get('eps_step', 0.01),
                'max_iter': kwargs.get('max_iter', 10)
            }
            print(f"Using native {attack_type.upper()} implementation: {attack_config}")
            return attack_config

    def generate_adversarial(self, img_array: np.ndarray, attack) -> np.ndarray:
        """
        Generate adversarial example using attack.

        Args:
            img_array: Clean image [H, W, C] in [0,1] range
            attack: Attack object (ART) or config dict (native)

        Returns:
            Adversarial image [H, W, C] in [0,1] range
        """
        if isinstance(attack, dict):
            # Native PyTorch implementation
            return self._generate_adversarial_native(img_array, attack)
        else:
            # ART implementation
            img_batch = np.expand_dims(img_array, axis=0)
            print("Generating adversarial example with ART...")
            adv_batch = attack.generate(x=img_batch)
            adv_array = adv_batch[0]
            adv_array = np.clip(adv_array, 0, 1)
            return adv_array

    def _generate_adversarial_native(self, img_array: np.ndarray, attack_config: dict) -> np.ndarray:
        """
        Native PyTorch implementation of adversarial attacks.
        """
        attack_type = attack_config['type']
        epsilon = attack_config['epsilon']

        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        original_tensor = img_tensor.clone()

        print(f"Generating adversarial example with native {attack_type.upper()}...")

        if attack_type == 'fgsm':
            adv_tensor = self._fgsm_native(img_tensor, epsilon)
        elif attack_type == 'pgd':
            adv_tensor = self._pgd_native(img_tensor, epsilon,
                                          attack_config['eps_step'],
                                          attack_config['max_iter'])
        elif attack_type == 'bim':
            adv_tensor = self._bim_native(img_tensor, epsilon,
                                          attack_config['eps_step'],
                                          attack_config['max_iter'])
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Convert back to numpy
        adv_array = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        adv_array = np.clip(adv_array, 0, 1)

        return adv_array

    def _fgsm_native(self, img_tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Native FGSM implementation."""
        img_tensor.requires_grad = True

        # Forward pass
        outputs = self.pytorch_model(img_tensor)

        # Loss: negative sum of all detection scores (we want to minimize detections)
        loss = -outputs[0]['scores'].sum()

        # Backward pass
        self.pytorch_model.zero_grad()
        loss.backward()

        # Create adversarial example
        data_grad = img_tensor.grad.data
        perturbed = img_tensor + epsilon * data_grad.sign()
        perturbed = torch.clamp(perturbed, 0, 1)

        return perturbed.detach()

    def _pgd_native(self, img_tensor: torch.Tensor, epsilon: float,
                    alpha: float, num_iter: int) -> torch.Tensor:
        """Native PGD implementation."""
        perturbed = img_tensor.clone().detach()
        original = img_tensor.clone().detach()

        for i in range(num_iter):
            perturbed.requires_grad = True

            outputs = self.pytorch_model(perturbed)
            loss = -outputs[0]['scores'].sum()

            self.pytorch_model.zero_grad()
            loss.backward()

            # Update
            data_grad = perturbed.grad.data
            perturbed = perturbed.detach() + alpha * data_grad.sign()

            # Project back to epsilon ball
            perturbation = torch.clamp(perturbed - original, -epsilon, epsilon)
            perturbed = torch.clamp(original + perturbation, 0, 1).detach()

        return perturbed

    def _bim_native(self, img_tensor: torch.Tensor, epsilon: float,
                    alpha: float, num_iter: int) -> torch.Tensor:
        """Native BIM (essentially same as PGD) implementation."""
        return self._pgd_native(img_tensor, epsilon, alpha, num_iter)

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def find_best_match(self, clean_box: np.ndarray, clean_label: int,
                        adv_dets: Dict) -> Tuple[int, float, int]:
        """
        Find best matching adversarial detection for a clean detection.

        Returns:
            Tuple of (match_index, iou, adv_label) or (-1, 0.0, -1) if no match
        """
        best_iou = 0.0
        best_idx = -1
        best_label = -1

        for i, adv_box in enumerate(adv_dets['boxes']):
            iou = self.calculate_iou(clean_box, adv_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
                best_label = adv_dets['labels'][i]

        return best_idx, best_iou, best_label

    def analyze_detection_errors(self, clean_dets: Dict, adv_dets: Dict) -> Dict:
        """
        Detailed analysis of adversarial detection errors.

        Categorizes each clean detection into:
        1. Correct detection (same class, good localization)
        2. Class misclassification (detected but wrong class)
        3. Localization error (correct class but poor box alignment)
        4. Complete miss (not detected at all)
        5. Mixed error (wrong class AND poor localization)
        6. False positives (adversarial detections with no corresponding clean object)

        Returns:
            Dictionary with detailed error analysis
        """
        num_clean = len(clean_dets['boxes'])

        # Initialize error categories
        errors = {
            'correct_detections': [],
            'class_misclassifications': [],
            'localization_errors': [],
            'complete_misses': [],
            'mixed_errors': [],  # Both class and localization errors
            'false_positives': [],  # Detections that don't match any clean object
        }

        # Track which adversarial detections have been matched
        matched_adv = set()

        # Analyze each clean detection
        for i in range(num_clean):
            clean_box = clean_dets['boxes'][i]
            clean_label = clean_dets['labels'][i]
            clean_score = clean_dets['scores'][i]

            # Find best matching adversarial detection
            best_idx, best_iou, adv_label = self.find_best_match(clean_box, clean_label, adv_dets)

            detection_info = {
                'clean_box': clean_box,
                'clean_label': clean_label,
                'clean_class': self.coco_names[clean_label],
                'clean_score': clean_score,
                'adv_box': adv_dets['boxes'][best_idx] if best_idx != -1 else None,
                'adv_label': adv_label if best_idx != -1 else None,
                'adv_class': self.coco_names[adv_label] if best_idx != -1 else None,
                'adv_score': adv_dets['scores'][best_idx] if best_idx != -1 else None,
                'iou': best_iou
            }

            # Categorize the error
            if best_idx == -1 or best_iou < self.iou_threshold:
                # No match found or IoU too low
                if best_idx == -1:
                    errors['complete_misses'].append(detection_info)
                else:
                    # There's a detection but IoU is below threshold
                    if adv_label == clean_label:
                        errors['localization_errors'].append(detection_info)
                    else:
                        errors['mixed_errors'].append(detection_info)
                    matched_adv.add(best_idx)
            else:
                # Good localization (IoU >= threshold)
                matched_adv.add(best_idx)
                if adv_label == clean_label:
                    # Correct detection
                    errors['correct_detections'].append(detection_info)
                else:
                    # Class misclassification
                    errors['class_misclassifications'].append(detection_info)

        # Find false positives (adversarial detections not matched to any clean detection)
        for i in range(len(adv_dets['boxes'])):
            if i not in matched_adv:
                fp_info = {
                    'adv_box': adv_dets['boxes'][i],
                    'adv_label': adv_dets['labels'][i],
                    'adv_class': self.coco_names[adv_dets['labels'][i]],
                    'adv_score': adv_dets['scores'][i],
                    'clean_box': None,
                    'clean_label': None,
                    'clean_class': None,
                    'clean_score': None,
                    'iou': 0.0
                }
                errors['false_positives'].append(fp_info)

        # Calculate statistics
        stats = {
            'total_clean_objects': num_clean,
            'correct_detections': len(errors['correct_detections']),
            'class_misclassifications': len(errors['class_misclassifications']),
            'localization_errors': len(errors['localization_errors']),
            'complete_misses': len(errors['complete_misses']),
            'mixed_errors': len(errors['mixed_errors']),
            'false_positives': len(errors['false_positives']),

            # Rates
            'correct_rate': len(errors['correct_detections']) / num_clean if num_clean > 0 else 0,
            'class_error_rate': len(errors['class_misclassifications']) / num_clean if num_clean > 0 else 0,
            'localization_error_rate': len(errors['localization_errors']) / num_clean if num_clean > 0 else 0,
            'miss_rate': len(errors['complete_misses']) / num_clean if num_clean > 0 else 0,
            'mixed_error_rate': len(errors['mixed_errors']) / num_clean if num_clean > 0 else 0,
        }

        return {
            'errors': errors,
            'stats': stats
        }

    def print_detailed_analysis(self, analysis: Dict):
        """Print detailed analysis of detection errors."""
        stats = analysis['stats']
        errors = analysis['errors']

        print(f"\n{'=' * 70}")
        print("DETAILED ERROR ANALYSIS")
        print(f"{'=' * 70}")

        print(f"\nTotal Clean Objects: {stats['total_clean_objects']}")
        print(f"\n{'─' * 70}")
        print("Error Breakdown:")
        print(f"{'─' * 70}")
        print(f"  ✓ Correct Detections:        {stats['correct_detections']:3d} ({stats['correct_rate']:.1%})")
        print(
            f"  ✗ Class Misclassifications:  {stats['class_misclassifications']:3d} ({stats['class_error_rate']:.1%})")
        print(
            f"  ✗ Localization Errors:       {stats['localization_errors']:3d} ({stats['localization_error_rate']:.1%})")
        print(f"  ✗ Complete Misses:           {stats['complete_misses']:3d} ({stats['miss_rate']:.1%})")
        print(f"  ✗ Mixed Errors:              {stats['mixed_errors']:3d} ({stats['mixed_error_rate']:.1%})")
        print(f"  + False Positives:           {stats['false_positives']:3d}")

        # Show examples of each error type
        print(f"\n{'─' * 70}")
        print("Error Examples:")
        print(f"{'─' * 70}")

        if errors['class_misclassifications']:
            print(f"\nClass Misclassifications (detected but wrong class):")
            for i, err in enumerate(errors['class_misclassifications'][:3]):  # Show first 3
                print(f"  [{i + 1}] {err['clean_class']} → {err['adv_class']} "
                      f"(IoU: {err['iou']:.2f}, Score: {err['clean_score']:.2f}→{err['adv_score']:.2f})")

        if errors['localization_errors']:
            print(f"\nLocalization Errors (correct class but poor box):")
            for i, err in enumerate(errors['localization_errors'][:3]):
                print(f"  [{i + 1}] {err['clean_class']} with IoU: {err['iou']:.2f} "
                      f"(Score: {err['clean_score']:.2f}→{err['adv_score']:.2f})")

        if errors['complete_misses']:
            print(f"\nComplete Misses (not detected):")
            for i, err in enumerate(errors['complete_misses'][:3]):
                print(f"  [{i + 1}] {err['clean_class']} (Score: {err['clean_score']:.2f})")

        if errors['mixed_errors']:
            print(f"\nMixed Errors (wrong class AND poor localization):")
            for i, err in enumerate(errors['mixed_errors'][:3]):
                print(f"  [{i + 1}] {err['clean_class']} → {err['adv_class']} "
                      f"(IoU: {err['iou']:.2f}, Score: {err['clean_score']:.2f}→{err['adv_score']:.2f})")

        if errors['false_positives']:
            print(f"\nFalse Positives (detected in adversarial but not in clean):")
            for i, err in enumerate(errors['false_positives'][:3]):
                print(f"  [{i + 1}] {err['adv_class']} detected (Score: {err['adv_score']:.2f})")

        print(f"\n{'=' * 70}")

    def visualize_detailed_comparison(self, clean_img: np.ndarray, clean_dets: Dict,
                                      adv_img: np.ndarray, adv_dets: Dict,
                                      analysis: Dict, title: str = "Robustness Test"):
        """Visualize with color-coded error types."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        errors = analysis['errors']

        # Define colors for each error type
        color_map = {
            'correct': 'green',
            'class_error': 'orange',
            'localization_error': 'yellow',
            'complete_miss': 'red',
            'mixed_error': 'purple',
            'false_positive': 'cyan'
        }

        # Clean image
        img_clean_pil = Image.fromarray((clean_img * 255).astype(np.uint8))
        draw_clean = ImageDraw.Draw(img_clean_pil)

        # Draw all clean detections in blue
        for box, label, score in zip(clean_dets['boxes'], clean_dets['labels'], clean_dets['scores']):
            draw_clean.rectangle(box.tolist(), outline='blue', width=3)
            text = f"{self.coco_names[label]}: {score:.2f}"
            draw_clean.text((box[0], max(0, box[1] - 15)), text, fill='blue')

        axes[0].imshow(img_clean_pil)
        axes[0].set_title(f"Clean Image ({len(clean_dets['boxes'])} detections)")
        axes[0].axis('off')

        # Adversarial image with color-coded errors
        img_adv_pil = Image.fromarray((adv_img * 255).astype(np.uint8))
        draw_adv = ImageDraw.Draw(img_adv_pil)

        # Draw correct detections
        for err in errors['correct_detections']:
            box = err['adv_box']
            draw_adv.rectangle(box.tolist(), outline=color_map['correct'], width=3)
            text = f"✓ {err['adv_class']}: {err['adv_score']:.2f}"
            draw_adv.text((box[0], max(0, box[1] - 15)), text, fill=color_map['correct'])

        # Draw class misclassifications
        for err in errors['class_misclassifications']:
            box = err['adv_box']
            draw_adv.rectangle(box.tolist(), outline=color_map['class_error'], width=3)
            text = f"CLASS: {err['clean_class']}→{err['adv_class']}"
            draw_adv.text((box[0], max(0, box[1] - 15)), text, fill=color_map['class_error'])

        # Draw localization errors
        for err in errors['localization_errors']:
            if err['adv_box'] is not None:
                box = err['adv_box']
                draw_adv.rectangle(box.tolist(), outline=color_map['localization_error'], width=3)
                text = f"LOC: {err['adv_class']} (IoU:{err['iou']:.2f})"
                draw_adv.text((box[0], max(0, box[1] - 15)), text, fill=color_map['localization_error'])

        # Draw mixed errors
        for err in errors['mixed_errors']:
            if err['adv_box'] is not None:
                box = err['adv_box']
                draw_adv.rectangle(box.tolist(), outline=color_map['mixed_error'], width=3)
                text = f"MIXED: {err['clean_class']}→{err['adv_class']}"
                draw_adv.text((box[0], max(0, box[1] - 15)), text, fill=color_map['mixed_error'])

        # Mark complete misses on clean boxes (shown as red X on adversarial image)
        for err in errors['complete_misses']:
            box = err['clean_box']
            draw_adv.rectangle(box.tolist(), outline=color_map['complete_miss'], width=2)
            draw_adv.text((box[0], max(0, box[1] - 15)), f"MISS: {err['clean_class']}",
                          fill=color_map['complete_miss'])

        # Draw false positives
        for err in errors['false_positives']:
            box = err['adv_box']
            draw_adv.rectangle(box.tolist(), outline=color_map['false_positive'], width=3)
            text = f"FP: {err['adv_class']}: {err['adv_score']:.2f}"
            draw_adv.text((box[0], max(0, box[1] - 15)), text, fill=color_map['false_positive'])

        axes[1].imshow(img_adv_pil)
        axes[1].set_title(f"Adversarial Image ({len(adv_dets['boxes'])} detections)")
        axes[1].axis('off')

        # Add legend
        legend_text = (
            f"Green=Correct | Orange=Class Error | Yellow=Localization Error\n"
            f"Red=Complete Miss | Purple=Mixed Error | Cyan=False Positive"
        )

        plt.suptitle(f"{title}\n{legend_text}", fontsize=12)
        plt.tight_layout()
        plt.show()

    def test_robustness(self, image_path: str, attack_type: str = 'fgsm',
                        epsilon: float = 0.03, visualize: bool = True, **attack_kwargs) -> Dict:
        """
        Complete robustness test with detailed error analysis.
        """
        print(f"\n{'=' * 70}")
        print(f"Testing robustness with {attack_type.upper()} attack (ε={epsilon})")
        print(f"{'=' * 70}")

        # Load and detect on clean image
        img_array, original_img = self.preprocess_image(image_path)
        print(f"Image shape: {img_array.shape}")
        clean_dets = self.detect_objects(img_array)
        print(f"Clean detections: {len(clean_dets['boxes'])}")

        # Create attack and generate adversarial example
        attack_kwargs['epsilon'] = epsilon
        attack = self.create_attack(attack_type, **attack_kwargs)
        adv_array = self.generate_adversarial(img_array, attack)

        # Detect on adversarial image
        adv_dets = self.detect_objects(adv_array)
        print(f"Adversarial detections: {len(adv_dets['boxes'])}")

        # Detailed error analysis
        analysis = self.analyze_detection_errors(clean_dets, adv_dets)

        # Print detailed analysis
        self.print_detailed_analysis(analysis)

        # Calculate perturbation magnitude
        perturbation = np.linalg.norm(adv_array - img_array)
        print(f"\nL2 Perturbation: {perturbation:.4f}")

        # Visualize
        if visualize:
            self.visualize_detailed_comparison(img_array, clean_dets, adv_array, adv_dets,
                                               analysis, title=f"{attack_type.upper()} Attack (ε={epsilon})")

        return {
            'analysis': analysis,
            'clean_detections': clean_dets,
            'adversarial_detections': adv_dets,
            'attack_type': attack_type,
            'epsilon': epsilon,
            'perturbation_l2': perturbation
        }

    def test_robustness_batch(self, image_paths: List[str], attack_type: str = 'fgsm',
                              epsilon: float = 0.03, visualize: bool = False, **attack_kwargs) -> List[Dict]:
        """
        Test robustness on a batch of images.

        Args:
            image_paths: List of image paths or URLs
            attack_type: Attack type to use
            epsilon: Attack strength
            visualize: Whether to visualize each image (can be slow for large batches)
            **attack_kwargs: Additional attack parameters

        Returns:
            List of result dictionaries, one per image
        """
        print(f"\n{'=' * 70}")
        print(f"BATCH ROBUSTNESS TESTING")
        print(f"{'=' * 70}")
        print(f"Testing {len(image_paths)} images with {attack_type.upper()} attack (ε={epsilon})")
        print(f"{'=' * 70}\n")

        all_results = []

        for i, image_path in enumerate(image_paths):
            print(f"\n[{i + 1}/{len(image_paths)}] Processing: {image_path}")
            print("-" * 70)

            try:
                result = self.test_robustness(
                    image_path,
                    attack_type=attack_type,
                    epsilon=epsilon,
                    visualize=visualize,
                    **attack_kwargs
                )
                result['image_path'] = image_path
                result['success'] = True
                all_results.append(result)

            except Exception as e:
                print(f"✗ Error processing image: {e}")
                all_results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': str(e)
                })

        # Print batch summary
        self._print_batch_summary(all_results)

        return all_results

    def _print_batch_summary(self, results: List[Dict]):
        """Print summary statistics for batch testing."""
        print(f"\n{'=' * 70}")
        print("BATCH TESTING SUMMARY")
        print(f"{'=' * 70}")

        successful = [r for r in results if r.get('success', False)]
        failed = len(results) - len(successful)

        print(f"\nTotal Images: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {failed}")

        if not successful:
            print("\nNo successful tests to summarize.")
            return

        # Aggregate statistics
        total_clean = sum(r['analysis']['stats']['total_clean_objects'] for r in successful)
        total_correct = sum(r['analysis']['stats']['correct_detections'] for r in successful)
        total_class_errors = sum(r['analysis']['stats']['class_misclassifications'] for r in successful)
        total_loc_errors = sum(r['analysis']['stats']['localization_errors'] for r in successful)
        total_misses = sum(r['analysis']['stats']['complete_misses'] for r in successful)
        total_mixed = sum(r['analysis']['stats']['mixed_errors'] for r in successful)
        total_false_positives = sum(r['analysis']['stats']['false_positives'] for r in successful)

        print(f"\n{'─' * 70}")
        print("AGGREGATE STATISTICS ACROSS ALL IMAGES")
        print(f"{'─' * 70}")
        print(f"Total Clean Objects: {total_clean}")
        print(f"  ✓ Correct Detections:        {total_correct:4d} ({total_correct / total_clean * 100:.1f}%)")
        print(f"  ✗ Class Misclassifications:  {total_class_errors:4d} ({total_class_errors / total_clean * 100:.1f}%)")
        print(f"  ✗ Localization Errors:       {total_loc_errors:4d} ({total_loc_errors / total_clean * 100:.1f}%)")
        print(f"  ✗ Complete Misses:           {total_misses:4d} ({total_misses / total_clean * 100:.1f}%)")
        print(f"  ✗ Mixed Errors:              {total_mixed:4d} ({total_mixed / total_clean * 100:.1f}%)")
        print(f"  + False Positives:           {total_false_positives:4d}")

        # Average perturbation
        avg_perturbation = np.mean([r['perturbation_l2'] for r in successful])
        print(f"\nAverage L2 Perturbation: {avg_perturbation:.4f}")

        # Per-image breakdown
        print(f"\n{'─' * 70}")
        print("PER-IMAGE BREAKDOWN")
        print(f"{'─' * 70}")
        for i, r in enumerate(successful):
            stats = r['analysis']['stats']
            img_name = r['image_path'].split('/')[-1] if '/' in r['image_path'] else r['image_path']
            print(f"\n[{i + 1}] {img_name}")
            print(f"    Clean objects: {stats['total_clean_objects']}")
            print(f"    Correct: {stats['correct_rate']:.1%} | "
                  f"Class errors: {stats['class_error_rate']:.1%} | "
                  f"Misses: {stats['miss_rate']:.1%} | "
                  f"Mixed errors: {stats['mixed_error_rate']:.1%} | "
                  f"False positives: {stats['false_positives']}")

        print(f"\n{'=' * 70}")