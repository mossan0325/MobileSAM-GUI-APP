# MobileSAM GUI App

Python application for interactive segmentation using MobileSAM ONNX models.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare ONNX models. Download encoder and decoder models from the MobileSAM-ONNX sample or convert them yourself. Place them in a directory, e.g., `model/vit_t_encoder.onnx` and `model/vit_t_decoder.onnx`.

## Usage

```bash
python app.py --image path/to/image.jpg --encoder path/to/vit_t_encoder.onnx --decoder path/to/vit_t_decoder.onnx
```

- Left click: add positive point
- Right click: add negative point
- Shift + drag left mouse: draw bounding box
- `r`: reset prompts
- `q`: quit

Segmentation results are drawn transparently on the image.
