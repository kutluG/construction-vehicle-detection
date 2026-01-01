import argparse
import cv2
from ultralytics import YOLO
import sys
import os

# COCO classes often found on construction sites or misclassified as such
RELEVANT_CLASSES = [2, 5, 7, 6, 8] # car, bus, truck, train, boat
# train/boat often catch large machinery like cranes/excavators
CLASS_MAPPING = {
    2: "Construction Vehicle", # car -> Construction Vehicle
    5: "Construction Vehicle", # bus -> Construction Vehicle
    7: "Construction Vehicle", # truck -> Construction Vehicle
    6: "Heavy Machinery",      # train -> Heavy Machinery
    8: "Heavy Machinery"       # boat -> Heavy Machinery (often excavator arms)
}

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Construction Vehicle Detection")
    parser.add_argument("source", type=str, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/y11x_10ep/weights/best.pt",
        help="YOLO model path (default: YOLO11x model)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (default: 1280)")
    parser.add_argument("--no-augment", action="store_true", help="Disable Test Time Augmentation (TTA)")
    parser.add_argument("--sahi", action="store_true", help="Enable Slicing Aided Hyper Inference (SAHI)")
    parser.add_argument("--slice-height", type=int, default=640, help="Slice height for SAHI (default: 640)")
    parser.add_argument("--slice-width", type=int, default=640, help="Slice width for SAHI (default: 640)")
    args = parser.parse_args()

    input_path = args.source
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        sys.exit(1)

    # Standardized list of detections: [{'label': str, 'conf': float, 'bbox': [x1, y1, x2, y2], 'class_id': int, 'orig_label': str}]
    raw_detections = []
    original_image = cv2.imread(input_path) 
    
    if original_image is None:
        print(f"Error: Could not read image at {input_path}")
        sys.exit(1)

    if args.sahi:
        # SAHI Inference
        print(f"Processing with SAHI (slice: {args.slice_height}x{args.slice_width})...")
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError:
            print("Error: SAHI not installed. Please run: pip install sahi")
            sys.exit(1)

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=args.model,
            confidence_threshold=args.conf,
            device="cpu" # or 'cuda' if available
        )
        
        result = get_sliced_prediction(
            input_path,
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        for obj in result.object_prediction_list:
            # obj.category contains 'id' and 'name'
            class_id = obj.category.id
            name = obj.category.name
            score = obj.score.value
            bbox = obj.bbox.to_xyxy() # [x1, y1, x2, y2]
            
            raw_detections.append({
                'class_id': class_id,
                'orig_label': name,
                'conf': score,
                'bbox': bbox
            })
            
    else:
        # Standard Inference
        print(f"Loading model {args.model}...")
        try:
            model = YOLO(args.model)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        # If this looks like a custom/fine-tuned model (e.g., single-class), don't apply COCO-only filtering.
        # Your fine-tuned excavator model has class_id=0, which was previously being filtered out.
        model_names = getattr(model, "names", {}) or {}
        is_coco_like = isinstance(model_names, dict) and len(model_names) >= 80
        allow_all_classes = not is_coco_like

        print(f"Processing {input_path} with imgsz={args.imgsz}, augment={not args.no_augment}...")
        results = model(input_path, conf=args.conf, imgsz=args.imgsz, augment=not args.no_augment)
        
        # Parse Ultralytics results
        result = results[0] # assuming single image
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            orig_label = model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            raw_detections.append({
                'class_id': class_id,
                'orig_label': orig_label,
                'conf': conf,
                'bbox': [x1, y1, x2, y2]
            })

    # --- POST-PROCESSING & NMS ---
    # We collect all valid boxes, map their labels, and THEN run Global NMS
    # because 'truck' (mapped to Construction Vehicle) and 'bus' (mapped to Construction Vehicle)
    # might overlap and need to be suppressed against each other.
    
    nms_boxes = []
    nms_scores = []
    final_labels = []     # Store mapped label
    final_orig_labels = [] # Store original label for debugging

    for det in raw_detections:
        class_id = det['class_id']
        conf = det['conf']
        orig_label = det['orig_label']
        x1, y1, x2, y2 = map(int, det['bbox'])
        
        # Map/filter label (COCO models only). For custom fine-tuned models, keep all classes.
        label = orig_label
        if 'allow_all_classes' in locals() and allow_all_classes:
            pass
        else:
            if class_id in CLASS_MAPPING:
                label = CLASS_MAPPING[class_id]
            elif class_id in RELEVANT_CLASSES:
                # Should be covered by mapping, but fallback
                pass
            else:
                # Irrelevant class, skip
                continue

        # Prepare for NMS
        # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h]
        w = x2 - x1
        h = y2 - y1
        nms_boxes.append([x1, y1, w, h])
        nms_scores.append(float(conf))
        final_labels.append(label)
        final_orig_labels.append(orig_label)

    # Ultralytics YOLO already applies NMS for standard inference.
    # Applying another (class-agnostic) NMS here can incorrectly suppress valid detections,
    # especially when a large box overlaps smaller ones (like in crowded scenes).
    # For SAHI, we still need an extra NMS to merge slice predictions.
    if args.sahi:
        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, score_threshold=args.conf, nms_threshold=0.45)
    else:
        indices = list(range(len(nms_boxes)))
    
    # VISUALIZATION
    img = original_image.copy()
    detected_count = 0
    
    # indices is sometimes a tuple or list of lists depending on cv2 version, handle gracefully
    if len(indices) > 0:
        chosen = indices.flatten() if hasattr(indices, "flatten") else indices
        for i in chosen:
            # Get data
            x1, y1, w, h = nms_boxes[i]
            x2, y2 = x1 + w, y1 + h
            label = final_labels[i]
            orig_label = final_orig_labels[i]
            conf = nms_scores[i]
            
            # Color: Green for CV, Red for HM (or something else?)
            # Let's keep distinct colors for the FINAL mapped classes
            if label == "Construction Vehicle":
                color = (0, 255, 0) # Green
            elif label == "Heavy Machinery":
                color = (0, 165, 255) # Orange-ish
            else:
                color = (0, 0, 255) # Red for unknown

            # Draw Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            print(f"- Detected: {label} (Original: {orig_label}, Conf: {conf:.2f})")
            detected_count += 1

    # Save result
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    prefix = "detected_sahi_" if args.sahi else "detected_"
    output_filename = os.path.join(output_dir, f"{prefix}{os.path.basename(input_path)}")
    cv2.imwrite(output_filename, img)
    print(f"Result saved to {output_filename}")

if __name__ == "__main__":
    main()
