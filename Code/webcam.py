# Inside project root
import video_main

# You can pick a face detector depending on Acc/speed requirements
emotion_recognizer = video_main.EmotionAnalysisVideo(
                        face_detector="mtcnn",
                        model_loc="models",
                        face_detection_threshold=0.0,
                    )
emotion_recognizer.emotion_analysis_video(
    video_path=None,
    detection_interval=1,
    save_output=False,
    preview=True,
    output_path="data/output.mp4",
    resize_scale=0.5,
)