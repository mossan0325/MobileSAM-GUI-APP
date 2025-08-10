import argparse
import cv2
import numpy as np
import onnxruntime as ort


# Preprocessing function from reference notebook

def preprocess_image(
    image,
    resize_width=1024,
    mean=np.array([123.675, 116.28, 103.53]),
    std=np.array([[58.395, 57.12, 57.375]])
):
    # BGR -> RGB
    temp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width, image_height = temp_image.shape[1], temp_image.shape[0]
    resize_image_width, resize_image_height = image_width, image_height

    if image_width > image_height:
        resize_image_width = resize_width
        resize_image_height = int((resize_width / image_width) * image_height)
    else:
        resize_image_width = resize_width
        resize_image_height = int((resize_width / image_height) * image_width)

    temp_image = cv2.resize(temp_image, (resize_image_width, resize_image_height))

    resize_info = {
        "image_width": image_width,
        "image_height": image_height,
        "resize_image_width": resize_image_width,
        "resize_image_height": resize_image_height,
    }

    temp_image = (temp_image - mean) / std
    temp_image = temp_image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    if resize_image_height < resize_image_width:
        temp_image = np.pad(
            temp_image,
            (
                (0, 0),
                (0, 0),
                (0, resize_width - resize_image_height),
                (0, 0)
            )
        )
    else:
        temp_image = np.pad(
            temp_image,
            (
                (0, 0),
                (0, 0),
                (0, 0),
                (0, resize_width - resize_image_width)
            )
        )

    return temp_image, resize_info


def preprocess_points(points, labels, resize_info):
    if len(points) == 0:
        return np.zeros((1, 0, 2), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)
    pts = np.array(points).astype(np.float32)
    lbls = np.array(labels).astype(np.float32)
    pts[..., 0] = pts[..., 0] * (resize_info['resize_image_width'] / resize_info['image_width'])
    pts[..., 1] = pts[..., 1] * (resize_info['resize_image_height'] / resize_info['image_height'])
    return pts[np.newaxis, :, :], lbls[np.newaxis, :]


def postprocess_mask(masks, resize_info):
    mask = masks[0][0]
    mask = (mask > 0).astype(np.uint8)
    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    mask = mask[:resize_info['resize_image_height'], :resize_info['resize_image_width']]
    mask = cv2.resize(
        mask,
        (resize_info['image_width'], resize_info['image_height']),
        interpolation=cv2.INTER_LINEAR,
    )
    return mask


def main():
    parser = argparse.ArgumentParser(description="MobileSAM ONNX GUI")
    parser.add_argument("--image", required=True, help="input image path")
    parser.add_argument("--encoder", required=True, help="encoder onnx model path")
    parser.add_argument("--decoder", required=True, help="decoder onnx model path")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError("Image not found")

    pre_img, resize_info = preprocess_image(image)
    enc_sess = ort.InferenceSession(args.encoder, providers=['CPUExecutionProvider'])
    embedding = enc_sess.run(None, {'image': pre_img})[0]
    dec_sess = ort.InferenceSession(args.decoder, providers=['CPUExecutionProvider'])

    positive_points = []
    negative_points = []
    boxes = []

    window = 'MobileSAM'
    cv2.namedWindow(window)

    drawing_box = False
    box_start = (0, 0)

    def run_decoder():
        if not (positive_points or negative_points or boxes):
            display = image.copy()
        else:
            pts = positive_points + negative_points
            lbls = [1] * len(positive_points) + [0] * len(negative_points)
            for x1, y1, x2, y2 in boxes:
                pts.extend([(x1, y1), (x2, y2)])
                lbls.extend([2, 3])
            input_point, input_label = preprocess_points(pts, lbls, resize_info)
            masks, scores, logits = dec_sess.run(None, {
                "image_embedding": embedding,
                "point_coords": input_point,
                "point_labels": input_label,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros(1, dtype=np.float32),
                "orig_im_size": np.array([resize_info['image_height'], resize_info['image_width']], dtype=np.float32)
            })
            mask = postprocess_mask(masks, resize_info)
            overlay = image.copy()
            overlay[mask > 0] = (0, 255, 0)
            display = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
        for pt in positive_points:
            cv2.circle(display, pt, 5, (255, 0, 0), -1, lineType=cv2.LINE_AA)
        for pt in negative_points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        return display

    def mouse(event, x, y, flags, param):
        nonlocal drawing_box, box_start
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                drawing_box = True
                box_start = (x, y)
            else:
                positive_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            negative_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
            display = run_decoder()
            cv2.rectangle(display, box_start, (x, y), (255, 255, 0), 2)
            cv2.imshow(window, display)
            return
        elif event == cv2.EVENT_LBUTTONUP and drawing_box:
            drawing_box = False
            boxes.append((box_start[0], box_start[1], x, y))
        cv2.imshow(window, run_decoder())

    cv2.setMouseCallback(window, mouse)
    cv2.imshow(window, image)
    print("left click: positive, right click: negative, Shift+drag: box, r: reset, q: quit")

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            positive_points.clear()
            negative_points.clear()
            boxes.clear()
            cv2.imshow(window, run_decoder())

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

