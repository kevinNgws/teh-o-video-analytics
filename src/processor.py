import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Generator, Tuple


class VideoProcessor:
    def __init__(self, output_dir: str = "output"):
        """
        Initializes the processor and ensures the output directory exists.
        """
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Color map for severity levels (R, G, B)
        self.severity_colors = {
            "high": (255, 0, 0),      # Red
            "medium": (255, 165, 0),  # Orange
            "low": (255, 255, 0),     # Yellow
            "unknown": (0, 255, 0)    # Green (Default)
        }

    def process_video_stream(self, video_path: str, interval_seconds: int = 1) -> Generator[Tuple[float, Image.Image], None, None]:
        """
        Generator that yields frames from a video at a specific time interval.

        Args:
            video_path: Path to the local video file.
            interval_seconds: How many seconds to skip between analyzing frames.
                              (e.g., 1 = analyze 1 frame per second).

        Yields:
            Tuple containing (timestamp_in_seconds, PIL_Image_Object)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        frame_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Only process frames at the specific interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps

                # Convert OpenCV BGR format to RGB (standard for AI models/PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                yield timestamp, pil_image

            frame_count += 1

        cap.release()

    def draw_detections(self, image: Image.Image, detections: List[Dict]) -> str:
        draw = ImageDraw.Draw(image)
        width, height = image.size

        # Load font... (same as before)
        try:
            font = ImageFont.truetype("Arial.ttf", size=20)
        except IOError:
            font = ImageFont.load_default()

        for det in detections:
            box = det.get("box", [])
            label = det.get("label", "Anomaly")
            severity = det.get("severity", "unknown").lower()

            if len(box) != 4:
                continue

            # --- CRITICAL: Coordinate Mapping ---
            # Prompt requested: [ymin, xmin, ymax, xmax]
            # ImageDraw expects: [x0, y0, x1, y1] (Left, Top, Right, Bottom)

            ymin_norm, xmin_norm, ymax_norm, xmax_norm = box

            # Denormalize (0-1000 -> pixels)
            left = (xmin_norm / 1000) * width
            top = (ymin_norm / 1000) * height
            right = (xmax_norm / 1000) * width
            bottom = (ymax_norm / 1000) * height

            # Safety clamp (keep box inside image)
            left = max(0, min(left, width))
            top = max(0, min(top, height))
            right = max(0, min(right, width))
            bottom = max(0, min(bottom, height))

            color = self.severity_colors.get(severity, (0, 255, 0))

            # Draw Box
            draw.rectangle([left, top, right, bottom], outline=color, width=4)

            # Draw Label
            text_str = f"{label} ({severity.title()})"

            # Calculate text background size
            text_bbox = draw.textbbox((left, top), text_str, font=font)

            # Draw label background
            draw.rectangle(text_bbox, fill=color)

            # Draw text
            draw.text((left, top), text_str, fill="black", font=font)

        filename = f"detection_{int(cv2.getTickCount())}.jpg"
        save_path = os.path.join(self.output_dir, filename)
        image.save(save_path)

        return save_path
