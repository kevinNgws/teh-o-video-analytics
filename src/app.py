import gradio as gr
import os
import time
from typing import List

# Import our custom modules
from analyzer import SafetyAnalyzer
from processor import VideoProcessor

# Configuration
VIDEO_DIR = "videos"
OUTPUT_DIR = "output"

# Ensure directories exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for the model to avoid reloading it on every click
# We initialize it as None and load it on startup or first run
analyzer = None


def load_model():
    """Lazy loader for the heavy AI model"""
    global analyzer
    if analyzer is None:
        print("Loading AI Model... check your terminal for download progress.")
        analyzer = SafetyAnalyzer()
    return analyzer


def get_video_files():
    """Scans the video directory for files"""
    files = [f for f in os.listdir(
        VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    if not files:
        return ["No videos found in 'videos/' folder"]
    return files


def safety_analysis_loop(video_name, user_prompt, progress=gr.Progress()):
    """
    The main logic loop. It yields updates to the UI step-by-step.
    """
    logs = []
    gallery_images = []

    # 1. Validation
    if not video_name or "No videos found" in video_name:
        logs.append("❌ Error: Please select a valid video.")
        yield "\n".join(logs), gallery_images
        return

    video_path = os.path.join(VIDEO_DIR, video_name)

    # 2. Load Model (if not loaded)
    logs.append("⏳ Initializing Safety AI (InternVL)... this may take a moment.")
    yield "\n".join(logs), gallery_images

    try:
        model = load_model()
        logs.append("✅ Model Loaded. Starting Video Processing...")
    except Exception as e:
        logs.append(f"❌ Model Load Failed: {str(e)}")
        yield "\n".join(logs), gallery_images
        return

    # 3. Initialize Processor
    processor = VideoProcessor(output_dir=OUTPUT_DIR)

    # 4. Stream Processing
    # We use the processor generator to get frames one by one
    frame_generator = processor.process_video_stream(
        video_path, interval_seconds=5)

    start_time = time.time()

    for timestamp, frame_img in frame_generator:
        # Update Log
        current_log = f"Processing timestamp: {timestamp:.2f}s..."
        logs.append(current_log)
        # Show last 10 logs to keep UI clean
        yield "\n".join(logs[-10:]), gallery_images

        # Analyze Frame
        try:
            detections = model.analyze_frame(frame_img, user_prompt)

            if detections:
                # If anomalies found, draw boxes and save
                logs.append(
                    f"⚠️ Anomaly Detected at {timestamp:.2f}s! Found {len(detections)} issues.")

                saved_path = processor.draw_detections(frame_img, detections)

                # Add to gallery (Gradio expects a list of paths or tuples)
                gallery_images.append(
                    (saved_path, f"{timestamp:.2f}s - {detections[0]['severity']} Severity"))

            else:
                logs.append(f"✅ No issues found at {timestamp:.2f}s.")

        except Exception as e:
            logs.append(f"⚠️ Error analyzing frame: {str(e)}")
            print(e)  # Print full trace to terminal

        # Yield results to update UI immediately
        yield "\n".join(logs[-10:]), gallery_images

    total_time = time.time() - start_time
    logs.append(f"🏁 Analysis Complete. Total time: {total_time:.2f}s")
    yield "\n".join(logs[-10:]), gallery_images

# --- Gradio UI Layout ---


with gr.Blocks(title="Safety Copilot POC (InternVL Local)") as demo:
    gr.Markdown("# 🦺 Safety Copilot: Local AI Analysis")
    gr.Markdown(
        "Run safety inspections on local videos using InternVL 2.5 on your M1 Mac.")

    with gr.Row():
        with gr.Column(scale=1):
            # Left Panel: Controls
            video_dropdown = gr.Dropdown(
                label="1. Select Video",
                choices=get_video_files(),
                value=get_video_files()[0] if get_video_files(
                ) and "No videos" not in get_video_files()[0] else None,
                interactive=True
            )
            refresh_btn = gr.Button("🔄 Refresh Video List", size="sm")

            default_safety_checks = (
                "1. MISSING PPE: Worker not wearing hard hat, high-vis vest, or safety shoes.\n"
                "2. CRANE SAFETY: Worker walking under a suspended or lifted load.\n"
                "3. OVERHEAD HAZARD: Heavy loads positioned unsecured above workers' heads.\n"
                "4. FALL RISK: Worker at height without a safety harness."
            )

            user_prompt = gr.Textbox(
                label="2. Safety Concern",
                value=default_safety_checks,
                lines=5,
                scale=3
            )

            analyze_btn = gr.Button(
                "🚀 Start Analysis", variant="primary", size="lg")

            logs_box = gr.TextArea(
                label="System Logs",
                placeholder="Waiting for start...",
                interactive=False,
                lines=10
            )

        with gr.Column(scale=2):
            # Right Panel: Results
            gallery = gr.Gallery(
                label="Detected Anomalies",
                columns=2,
                height=800,
                object_fit="contain"
            )

    # Event Wiring
    refresh_btn.click(
        fn=lambda: gr.update(choices=get_video_files()),
        outputs=video_dropdown
    )

    analyze_btn.click(
        fn=safety_analysis_loop,
        inputs=[video_dropdown, user_prompt],
        outputs=[logs_box, gallery]
    )

if __name__ == "__main__":
    # Launch with share=False for local only
    demo.launch(server_name="0.0.0.0", server_port=7860)
