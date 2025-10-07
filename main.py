import argparse
import os
from src.main import run_analysis

# --- Default Configuration ---
# This replaces the old config.json file
DEFAULT_CONFIG = {
    "frame_skip": 10,
    "inactive_game_frame_limit": 50,
    "min_player_positions": 15,
    "pixels_to_meters": 0.1,  # Example value, should be calibrated
    "team_clustering_sample_frames": 20,
    "team_names": {
        "0": "Team A",
        "1": "Team B"
    },
    "ocr_interval": 25
}

def main():
    parser = argparse.ArgumentParser(description="Football Match Analysis")
    parser.add_argument('--video', required=True, help="Path to the input video file.")
    parser.add_argument('--output', default='output', help="Directory to save the results.")
    parser.add_argument('--model', default='models/yolov8n.pt', help="Path to the YOLO model file.")
    parser.add_argument('--llm', action='store_true', help="Enable tactical report generation using an LLM (requires OPENAI_API_KEY).")

    args = parser.parse_args()

    # --- Validate paths ---
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at '{args.video}'")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at '{args.model}'")
        return

    # --- Run the analysis ---
    print(f"Starting analysis for video: {args.video}")
    print(f"Using model: {args.model}")
    print(f"Output will be saved to: {args.output}")
    if args.llm:
        print("LLM tactical report generation is ENABLED.")

    try:
        run_analysis(
            video_path=args.video,
            output_dir=args.output,
            model_path=args.model,
            config=DEFAULT_CONFIG,
            generate_llm_report=args.llm
        )
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == '__main__':
    main()