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
        """
        Draws bounding boxes and labels on the image and saves it to disk.

        Args:
            image: The original PIL image.
            detections: List of dictionaries. Example format:
                        [
                            {
                                "box": [ymin, xmin, ymax, xmax], # Normalized 0-1000
                                "label": "No Helmet",
                                "severity": "high"
                            }
                        ]
        
        Returns:
            str: Path to the saved image file.
        """
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Try to load a default font, strictly optional fallback if system font fails
        try:
            font = ImageFont.truetype("Arial.ttf", size=20)
        except IOError:
            font = ImageFont.load_default()

        found_anomaly = False

        for det in detections:
            found_anomaly = True
            box = det.get("box", [])
            label = det.get("label", "Anomaly")
            severity = det.get("severity", "unknown").lower()
            
            # Skip if box is malformed
            if len(box) != 4:
                continue

            # Unpack and denormalize coordinates (assuming 0-1000 scale from InternVL/Gemini)
            # Standard Format: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = box
            
            left = (xmin / 1000) * width
            top = (ymin / 1000) * height
            right = (xmax / 1000) * width
            bottom = (ymax / 1000) * height

            color = self.severity_colors.get(severity, (0, 255, 0))

            # Draw the Box (thick lines for visibility)
            draw.rectangle([left, top, right, bottom], outline=color, width=4)

            # Draw the Label Background
            text_bbox = draw.textbbox((left, top), f"{label} ({severity})", font=font)
            draw.rectangle(text_bbox, fill=color)

            # Draw the Text
            draw.text((left, top), f"{label} ({severity})", fill="black", font=font)

        # Save result
        filename = f"detection_{int(cv2.getTickCount())}.jpg"
        save_path = os.path.join(self.output_dir, filename)
        
        # If no anomalies, we might still want to return the path or None
        # For this POC, we return path regardless, but you can filter here.
        image.save(save_path)
        
        return save_path