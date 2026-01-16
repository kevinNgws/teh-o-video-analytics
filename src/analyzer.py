import torch
import re
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Configuration constants
# We use the 4B model as it offers the best balance of speed/quality for M1 chips (16GB RAM recommended)
# If you have 8GB RAM, change this to "OpenGVLab/InternVL2_5-2B"
MODEL_PATH = "OpenGVLab/InternVL2_5-8B"


class SafetyAnalyzer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initializing SafetyAnalyzer on {self.device}...")
        print(
            f"Loading model: {MODEL_PATH}. This will download ~8GB of data on first run.")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False
        )

        # Load Model (Using bfloat16 for Apple Silicon efficiency)
        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device).eval()

        print("Model loaded successfully.")

    def build_transform(self, input_size):
        """
        Standard image transformation pipeline required by InternVL.
        """
        MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_target_aspect_ratio(self, aspect_ratio, target_ratios, original_width, original_height, image_size):
        """
        Helper to handle dynamic resolution processing (InternVL specific logic).
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = original_width * original_height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        """
        Splits high-res images into tiles so the model sees details.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate best grid
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self.find_target_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Split image into blocks
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) > 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def analyze_frame(self, image: Image.Image, user_prompt: str):
        """
        Runs the model on a single frame.
        """
        # 1. Preprocess Image
        # We load a dynamic grid of tiles + 1 thumbnail.
        pixel_values = [self.build_transform(input_size=448)(
            img) for img in self.dynamic_preprocess(image)]
        pixel_values = torch.stack(pixel_values).to(
            torch.bfloat16).to(self.device)

        # 2. Construct Prompt
        # We instruct the model to act as a Safety Officer and look for specific bounding boxes.
        system_prompt = (
            "You are a Safety Officer. Analyze the image based on the user's concern. "
            "If a safety anomaly is found, identify the object, determine severity (High, Medium, or Low), "
            "and provide the bounding box in the format [[xmin, ymin, xmax, ymax]]."
        )

        question = f"{system_prompt}\nUser Concern: {user_prompt}\nResponse:"

        # InternVL format for multi-image input
        num_patches_list = [pixel_values.size(0)]

        # 3. Generation
        generation_config = dict(max_new_tokens=512, do_sample=False)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list
        )

        # 4. Parse Response
        return self.parse_response(response)

    def parse_response(self, text: str):
        """
        Extracts structured data from the model's text output using Regex.
        Example Text: "I see a worker without a helmet. Severity: High. Box: [[100, 200, 300, 400]]"
        """
        results = []

        # Heuristic to determine severity if explicitly stated
        severity_match = re.search(r'(High|Medium|Low)', text, re.IGNORECASE)
        # Default to Medium if unsure
        severity = severity_match.group(0) if severity_match else "Medium"

        # Extract Bounding Boxes
        # Pattern looks for [[x1, y1, x2, y2]]
        box_pattern = r'\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
        matches = re.findall(box_pattern, text)

        for match in matches:
            # InternVL typically returns 0-1000 normalized coords
            # Note: InternVL format is typically [xmin, ymin, xmax, ymax]
            # We convert to int
            box = [int(c) for c in match]

            # Simple check to ensure box is valid
            if len(box) == 4:
                # We assume the whole text applies to all boxes found for now
                results.append({
                    "label": "Safety Anomaly",  # In a real app, we'd parse the specific object name
                    "severity": severity,
                    # SWAP to [ymin, xmin, ymax, xmax] for processor.py
                    "box": [box[1], box[0], box[3], box[2]]
                })

        return results
