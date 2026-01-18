import json
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
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

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

        # --- JSON SYSTEM PROMPT ---
        system_prompt = (
            "You are a Senior Safety Officer. Analyze the image for the following concern: '{user_prompt}'.\n"
            "Rules:\n"
            "1. If NO safety anomalies are found, return an empty JSON array: []\n"
            "2. If anomalies are found, return a valid JSON array of objects.\n"
            "3. Each object must have these keys:\n"
            "   - 'description': Detailed description of the hazard.\n"
            "   - 'severity': 'High', 'Medium', or 'Low'.\n"
            "   - 'box': A list of 4 integers [ymin, xmin, ymax, xmax] normalized 0-1000.\n"
            "4. Do NOT output markdown code blocks (like ```json). Just the raw JSON string.\n"
            "5. STRICTLY use DOUBLE QUOTES (\") for all keys and string values. Do NOT use single quotes.\n"
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

        return self.parse_json_response(response)

    def parse_json_response(self, text: str):
        """
         robust JSON parser that handles common LLM formatting errors.
        """
        print(f"DEBUG: Raw Model Output -> {text}")  # Helpful for debugging

        try:
            # 1. Clean the text
            # Sometimes models wrap output in ```json ... ``` or add extra text.
            # We look for the first '[' and the last ']'
            start_index = text.find('[')
            end_index = text.rfind(']')

            if start_index == -1 or end_index == -1:
                # No JSON array found implies safe or malformed
                return []

            json_str = text[start_index:end_index+1]

            # 2. Parse
            data = json.loads(json_str)

            # 3. Validation & Cleanup
            valid_results = []
            for item in data:
                # Ensure keys exist
                if 'box' in item and len(item['box']) == 4:
                    # InternVL might drift on coordinate ordering despite prompts.
                    # If you find boxes look rotated, swap x/y here.
                    # Current prompt asks for [ymin, xmin, ymax, xmax].
                    # We pass it through directly.
                    valid_results.append({
                        # Processor expects 'label'
                        "label": item.get('description', 'Unknown Anomaly'),
                        "severity": item.get('severity', 'Medium'),
                        "box": item['box']
                    })

            return valid_results

        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            return []
