import argparse
from pathlib import Path

from ultralytics import YOLO


def train() -> None:
    parser = argparse.ArgumentParser(description="Train/fine-tune Ultralytics YOLO (v8 or v11)")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11s.pt",
        help="Pretrained weights to start from (e.g. yolo11s.pt, yolov8n.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset yaml (defaults to data_split.yaml if it exists, else data.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto)")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. 0, 0,1, cpu)")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project dir")
    parser.add_argument("--name", type=str, default="train", help="Run name")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots")
    parser.add_argument(
        "--export",
        type=str,
        default="onnx",
        help="Export format after training (e.g. onnx, torchscript). Use 'none' to skip.",
    )
    args = parser.parse_args()

    if args.data is None:
        args.data = "data_split.yaml" if Path("data_split.yaml").exists() else "data.yaml"

    model = YOLO(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        plots=not args.no_plots,
    )

    if args.export and args.export.lower() != "none":
        model.export(format=args.export)


if __name__ == "__main__":
    train()
