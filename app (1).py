import gradio as gr
from ultralytics import YOLO
import supervision as sv

    
box_annotator = sv.BoxAnnotator()
category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}



def yolov10_inference(image, model_id, image_size, conf_threshold, iou_threshold):
    model = YOLO(f"{model_id}.pt")
    results = model(source=image, imgsz=image_size, iou=iou_threshold, conf=conf_threshold, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    labels = [
        f"{category_dict[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = box_annotator.annotate(image, detections=detections, labels=labels)

    return annotated_image

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image")
                
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov10n",
                        "yolov10s",
                        "yolov10m",
                        "yolov10b",
                        "yolov10l",
                        "yolov10x",
                    ],
                    value="yolov10m",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.25,
                )
                iou_threshold = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.45,
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Annotated Image")

        yolov10_infer.click(
            fn=yolov10_inference,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_image],
        )

        gr.Examples(
            examples=[
                [
                    "dog.jpeg",
                    "yolov10x",
                    640,
                    0.25,
                    0.45,
                ],
                [
                    "huggingface.jpg",
                    "yolov10m",
                    640,
                    0.25,
                    0.45,
                ],
                [
                    "zidane.jpg",
                    "yolov10b",
                    640,
                    0.25,
                    0.45,
                ],
            ],
            fn=yolov10_inference,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
                iou_threshold,
            ],
            outputs=[output_image],
            cache_examples="lazy",
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv10: Real-Time End-to-End Object Detection
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Follow me for more!
        <a href='https://twitter.com/kadirnar_ai' target='_blank'>Twitter</a> | <a href='https://github.com/kadirnar' target='_blank'>Github</a> | <a href='https://www.linkedin.com/in/kadir-nar/' target='_blank'>Linkedin</a>  | <a href='https://www.huggingface.co/kadirnar/' target='_blank'>HuggingFace</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()

gradio_app.launch(debug=True)