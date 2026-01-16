import torch
import re
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Configuration
MODEL_PATH = "OpenGVLab/InternVL3-8B"  # or 1B/2B depending on RAM


class SafetyAnalyzer:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Initializing SafetyAnalyzer on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False
        )

        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device).eval()

    def build_transform(self, input_size):
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

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        """
        Standard InternVL dynamic tiling preprocessing.
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find best aspect ratio (simplified for brevity)
        best_ratio = min(target_ratios, key=lambda r: abs(
            aspect_ratio - (r[0] / r[1])))

        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) > 1:
            processed_images.append(image.resize((image_size, image_size)))

        return processed_images

    def analyze_frame(self, image: Image.Image, user_prompt: str):
        pixel_values = [self.build_transform(input_size=448)(
            img) for img in self.dynamic_preprocess(image)]
        pixel_values = torch.stack(pixel_values).to(
            torch.bfloat16).to(self.device)

        # --- ENHANCED PROMPT ---
        # 1. We strictly define the output format.
        # 2. We explicitly ask for 0-1000 normalization.
        # 3. We define the severity levels.
        system_prompt = (
            "You are a Senior Safety Officer. Analyze the image for the following concern: '{user_prompt}'.\n"
            "Rules:\n"
            "1. If NO safety anomalies are found, output EXACTLY: 'Result: Safe'.\n"
            "2. If anomalies are found, list them strictly in this format:\n"
            "   Anomaly: [Detailed description of the hazard]\n"
            "   Severity: [High (Life threatening) | Medium (Violation) | Low (Caution)]\n"
            "   Box: [[xmin, ymin, xmax, ymax]]\n"
            "3. Coordinates must be normalized 0-1000.\n"
            "Response:"
        )

        num_patches_list = [pixel_values.size(0)]
        generation_config = dict(max_new_tokens=512, do_sample=False)

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            system_prompt.format(user_prompt=user_prompt),
            generation_config,
            num_patches_list=num_patches_list
        )

        return self.parse_response(response)

    def parse_response(self, text: str):
        """
        Parses the strict output format.
        """
        # If model says safe, return nothing (Processor will skip drawing)
        if "Result: Safe" in text or "No anomalies" in text:
            return []

        results = []

        # Regex to find blocks of Anomaly ... Box
        # We look for the pattern defined in the prompt
        pattern = r"Anomaly:\s*(.*?)\n\s*Severity:\s*(.*?)\n\s*Box:\s*(\[\[.*?\]\])"
        matches = re.findall(pattern, text, re.DOTALL)

        for description, severity_raw, box_raw in matches:
            # Clean severity
            severity = "Low"
            if "High" in severity_raw:
                severity = "High"
            elif "Medium" in severity_raw:
                severity = "Medium"

            # Parse Box Integers
            # InternVL outputs [xmin, ymin, xmax, ymax] by default
            coords = re.findall(r"\d+", box_raw)
            if len(coords) == 4:
                x1, y1, x2, y2 = map(int, coords)

                # USER REQUIREMENT 4 & 6:
                # Return [ymin, xmin, ymax, xmax] format.
                # Only the list of 4 integers.
                final_box = [y1, x1, y2, x2]

                results.append({
                    "label": description.strip(),
                    "severity": severity,
                    "box": final_box
                })

        return results
