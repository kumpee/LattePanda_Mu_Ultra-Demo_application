# demo_hand_009.py
# Two-hand image adjustment + face/pose/hand landmarks + object detection + YOLO26 depth.
# Dependencies: pip install opencv-python numpy mediapipe ultralytics openvino
# MediaPipe Tasks models are downloaded automatically on first run.

import argparse
import os
import shutil
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np
import openvino as ov

MODEL_URLS = {
    'hand': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    'face': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'pose': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
}
RESOLUTIONS = [(640, 480), (960, 540), (1280, 720), (320, 240)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
              (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
              (27, 29), (29, 31), (28, 30), (30, 32)]


def get_model(name, directory):
    # Keep the original local filenames for compatibility with version 006.
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name + '_landmarker.task')
    if os.path.isfile(path) and os.path.getsize(path) > 1024:
        return path

    # Download to a temporary file so failures never leave a corrupt .task file.
    tmp = path + '.download'
    print('Downloading MediaPipe model:', name)
    try:
        urllib.request.urlretrieve(MODEL_URLS[name], tmp)
        if os.path.getsize(tmp) <= 1024:
            raise RuntimeError('Downloaded MediaPipe model is unexpectedly small: ' + name)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def new_hand_state():
    return {'active': False, 'anchor': (0.0, 0.0), 'values': (0.0, 1.0)}


def apply_adjustments(frame, hue, saturation, contrast, brightness):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + round(hue)) % 180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip((output.astype(np.float32) - 128) * contrast + 128 + brightness,
                   0, 255).astype(np.uint8)


def draw_bar(img, x, y, width, label, value, lo, hi, color):
    cv2.putText(img, f'{label}: {value:+.1f}' if lo < 0 else f'{label}: {value:.2f}',
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (245, 245, 245), 1, cv2.LINE_AA)
    y0 = y + 5
    cv2.rectangle(img, (x, y0), (x + width, y0 + 7), (75, 75, 75), -1)
    fill = int(width * np.clip((value - lo) / (hi - lo), 0, 1))
    cv2.rectangle(img, (x, y0), (x + fill, y0 + 7), color, -1)
    cv2.rectangle(img, (x, y0), (x + width, y0 + 7), (190, 190, 190), 1)


def draw_hud(img, values, states, fps, res, depth_on, detect_on):
    h, w = img.shape[:2]
    panel_w = min(w - 8, 354)
    panel_h = min(h - 8, 172)
    roi = img[4:4 + panel_h, 4:4 + panel_w]
    cv2.addWeighted(np.full_like(roi, (22, 22, 22)), 0.65, roi, 0.35, 0, dst=roi)
    hue, sat, contrast, bright = values
    cv2.putText(img, f'{fps:.1f} FPS  {res[0]}x{res[1]}  D:{int(depth_on)} O:{int(detect_on)}',
                (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    if panel_h >= 150:
        draw_bar(img, 12, 41, min(panel_w - 20, 155), 'Hue', hue, -90, 90, (80, 235, 80))
        draw_bar(img, 12, 73, min(panel_w - 20, 155), 'Saturation', sat, 0, 3, (80, 235, 80))
        draw_bar(img, 183, 41, max(40, panel_w - 193), 'Contrast', contrast, 0.2, 3, (255, 185, 65))
        draw_bar(img, 183, 73, max(40, panel_w - 193), 'Brightness', bright, -100, 100, (255, 185, 65))
        cv2.putText(img, f'L: {"GRAB" if states["Left"]["active"] else "FREE"}    R: {"GRAB" if states["Right"]["active"] else "FREE"}',
                    (12, 109), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 245, 245), 1)
        cv2.putText(img, 'SPACE res | D/O toggle | C/G/N det',
                    (12, 131), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1)
        cv2.putText(img, 'Shift+C/G/N depth | R reset S save Q quit',
                    (12, 151), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


def points_px(points, w, h):
    return [(int(p.x * w), int(p.y * h)) for p in points]


def draw_edges(img, pts, edges, color, radius=2):
    for a, b in edges:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def resolve_yolo_model(model_path, model_dir):
    # Prefer previously exported OpenVINO models or local weights.
    # Download only missing official YOLO26 checkpoints.
    requested = Path(model_path).expanduser()
    base = Path(model_dir).expanduser()
    candidates = [requested, base / requested]
    if not requested.suffix:
        candidates += [Path(str(requested) + '_openvino_model'),
                       base / (str(requested) + '_openvino_model'),
                       Path(str(requested) + '.pt'),
                       base / (str(requested) + '.pt')]
    for path in candidates:
        if path.is_file() and path.suffix.lower() in ('.pt', '.onnx', '.xml'):
            return str(path.resolve())
        if path.is_dir() and list(path.glob('*.xml')):
            return str(path.resolve())

    # Only official model names can be downloaded; custom models require a file.
    import re
    filename = requested.name
    if not filename.endswith('.pt'):
        filename += '.pt'
    if not re.fullmatch(r'yolo26[nsm lx](?:-depth)?\.pt'.replace(' ', ''), filename):
        raise FileNotFoundError('Model not found locally: ' + str(requested) +
                                '. Automatic download requires an official YOLO26 model name.')
    from ultralytics.utils.downloads import attempt_download_asset
    base.mkdir(parents=True, exist_ok=True)
    target = base / filename
    print('Downloading missing YOLO26 model:', filename)
    downloaded = Path(attempt_download_asset(filename)).resolve()
    if downloaded != target.resolve():
        shutil.copy2(downloaded, target)
    if not target.is_file():
        raise FileNotFoundError('YOLO model download failed: ' + filename)
    return str(target.resolve())



def prepare_yolo_model(local_path, imgsz, task=None):
    # Always use an OpenVINO IR, exactly as the working tracking benchmark does.
    from ultralytics import YOLO
    path = Path(local_path).resolve()
    if path.is_dir():
        xmls = sorted(path.glob('*.xml'))
        if xmls:
            return xmls[0]
    if path.suffix.lower() == '.xml':
        return path
    if path.suffix.lower() != '.pt':
        raise RuntimeError('OpenVINO export needs a .pt model: ' + str(path))
    export_dir = path.with_name(path.stem + '_openvino_model')
    xmls = sorted(export_dir.glob('*.xml')) if export_dir.exists() else []
    if xmls:
        print('Reusing OpenVINO export:', xmls[0])
        return xmls[0]
    print(f'Exporting {path.name} to OpenVINO FP16 ({imgsz}x{imgsz})...')
    model = YOLO(str(path), task=task) if task else YOLO(str(path))
    exported = Path(model.export(format='openvino', imgsz=imgsz, batch=1,
                                 dynamic=False, half=True, device='cpu', nms=None))
    xmls = sorted(exported.glob('*.xml')) if exported.is_dir() else [exported]
    if not xmls or not xmls[0].is_file():
        raise RuntimeError('OpenVINO XML not found after export: ' + str(exported))
    return xmls[0].resolve()


def load_ov(core, xml, device):
    # Match test_yolo26_object_tracking_002.py: explicit OpenVINO compilation.
    if device not in core.available_devices:
        raise RuntimeError(f'{device} unavailable; available: {core.available_devices}')
    compiled = core.compile_model(str(xml), device, {'PERFORMANCE_HINT': 'LATENCY'})
    shape = tuple(int(v) for v in compiled.input(0).shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise RuntimeError(f'Unexpected OpenVINO input shape: {shape}')
    if compiled.input(0).element_type == ov.Type.f16:
        dtype = np.float16
    elif compiled.input(0).element_type == ov.Type.f32:
        dtype = np.float32
    else:
        raise RuntimeError('Unsupported OpenVINO input datatype')
    inp = np.empty(shape, dtype=dtype)
    request = compiled.create_infer_request()
    request.set_input_tensor(ov.Tensor(inp, shared_memory=True))
    print(f'OpenVINO {xml.parent.name} on {device}: input={shape}, outputs=',
          [tuple(x.shape) for x in compiled.outputs])
    return {'req': request, 'inp': inp, 'shape': shape,
            'output': [request.get_output_tensor(i).data for i in range(len(compiled.outputs))]}


def ov_preprocess(frame, model):
    # Equivalent direct OpenVINO preprocessing to the supplied benchmark script.
    h, w = model['shape'][2:]
    resized = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    model['inp'][0] = (rgb.transpose(2, 0, 1) / 255.0).astype(model['inp'].dtype)


def decode_detections(raw, frame_shape, confidence=0.25, iou=0.5, max_det=100, input_size=(640, 640)):
    # YOLO26 end-to-end (1,N,6), or raw (1,4+nc,N) output.
    arr = np.asarray(raw)
    if arr.ndim != 3 or arr.shape[0] != 1:
        raise RuntimeError(f'Unexpected detection output {arr.shape}')
    a = arr[0]
    h, w = frame_shape[:2]
    # End-to-end output uses xyxy + confidence + class.
    if a.shape[1] == 6:
        det = a[a[:, 4] >= confidence].astype(np.float32, copy=True)
        det = det[np.argsort(-det[:, 4])[:max_det]]
    else:
        pred = a.T if a.shape[0] < a.shape[1] else a
        if pred.shape[1] < 6:
            raise RuntimeError(f'Unexpected YOLO26 detection tensor: {a.shape}')
        scores = pred[:, 4:]
        classes = np.argmax(scores, axis=1)
        conf = scores[np.arange(len(scores)), classes]
        valid = np.flatnonzero(conf >= confidence)
        if not len(valid):
            return np.empty((0, 6), np.float32)
        if len(valid) > 2000:
            valid = valid[np.argpartition(conf[valid], -2000)[-2000:]]
        xywh = pred[valid, :4].astype(np.float32)
        xyxy = np.column_stack((xywh[:, 0] - xywh[:, 2] / 2,
                                xywh[:, 1] - xywh[:, 3] / 2,
                                xywh[:, 0] + xywh[:, 2] / 2,
                                xywh[:, 1] + xywh[:, 3] / 2))
        nmsboxes = np.column_stack((xyxy[:, 0], xyxy[:, 1],
                                    xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]))
        offsets = classes[valid] * 10000.0
        nmsboxes[:, 0] += offsets
        indices = cv2.dnn.NMSBoxes(nmsboxes.tolist(), conf[valid].tolist(),
                                   confidence, iou, top_k=max_det)
        indices = np.asarray(indices).reshape(-1).astype(int)
        if not len(indices):
            return np.empty((0, 6), np.float32)
        det = np.column_stack((xyxy[indices], conf[valid[indices]],
                               classes[valid[indices]])).astype(np.float32)
    if not len(det):
        return np.empty((0, 6), np.float32)
    # This export uses direct square resizing, as in the working tracking script.
    det[:, [0, 2]] *= w / input_size[0]
    det[:, [1, 3]] *= h / input_size[1]
    det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, w - 1)
    det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, h - 1)
    return det[(det[:, 2] > det[:, 0]) & (det[:, 3] > det[:, 1])]


def ov_depth_to_map(model, width, height):
    raw = np.asarray(model['output'][0])
    # Typically Bx1xHxW or BxHxW; depth is output in meters.
    depth = np.squeeze(raw).astype(np.float32)
    if depth.ndim != 2:
        raise RuntimeError(f'Unexpected depth output shape: {raw.shape}')
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)


def depth_to_color(depth_data, size):
    # YOLO26 returns a metric depth map; display inverse depth (near = warm).
    w, h = size
    depth = np.asarray(depth_data, dtype=np.float32)
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((h, w, 3), np.uint8)
    inv = np.zeros_like(depth)
    inv[valid] = 1.0 / np.maximum(depth[valid], 1e-5)
    lo, hi = np.percentile(inv[valid], (2, 98))
    gray = np.uint8(np.clip((inv - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255))
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--pinch-on', type=float, default=0.30)
    parser.add_argument('--pinch-off', type=float, default=0.42)
    parser.add_argument('--yolo-model', default='yolo26s',
                        help='Local YOLO26 .pt/.onnx or OpenVINO export directory (or model name)')
    parser.add_argument('--yolo-dir', default='models',
                        help='Directory containing existing YOLO model files/exports')
    parser.add_argument('--yolo-device', default='CPU', choices=['CPU', 'GPU', 'NPU'],
                        help='YOLO device; GPU/NPU automatically exports local .pt models')
    parser.add_argument('--yolo-conf', type=float, default=0.25)
    parser.add_argument('--yolo-imgsz', type=int, default=640)
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'NPU'],
                        help='YOLO26 depth device; GPU/NPU automatically exports local .pt models')
    parser.add_argument('--depth-model', default='yolo26n-depth',
                        help='YOLO26 depth model: local file/export or official model name')
    parser.add_argument('--depth-imgsz', type=int, default=768)
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument('--no-objects', action='store_true')
    parser.add_argument('--detect-every', type=int, default=2)
    parser.add_argument('--depth-every', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    args = parser.parse_args()
    args.yolo_model = args.yolo_model.strip()
    if args.pinch_off <= args.pinch_on:
        parser.error('--pinch-off must be greater than --pinch-on')

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    def options(name, cls, **kwargs):
        return cls(base_options=python.BaseOptions(model_asset_path=get_model(name, args.model_dir)),
                   running_mode=vision.RunningMode.VIDEO, **kwargs)

    hand_opts = options('hand', vision.HandLandmarkerOptions, num_hands=2)
    face_opts = options('face', vision.FaceLandmarkerOptions, num_faces=1)
    pose_opts = options('pose', vision.PoseLandmarkerOptions, num_poses=1)

    core = ov.Core()
    print('OpenVINO:', ov.__version__, '| Available devices:', core.available_devices)
    # Compile/cache independent detection and depth inference requests per device.
    det_xml = depth_xml = None
    if not args.no_objects:
        det_path = resolve_yolo_model(args.yolo_model, args.yolo_dir)
        det_xml = prepare_yolo_model(det_path, args.yolo_imgsz)
    if not args.no_depth:
        depth_path = resolve_yolo_model(args.depth_model, args.yolo_dir)
        depth_xml = prepare_yolo_model(depth_path, args.depth_imgsz, task='depth')
    cache = {}

    def activate(kind, xml, device):
        key = (kind, str(xml), device)
        if key not in cache:
            mdl = load_ov(core, xml, device)
            mdl['inp'].fill(0)
            for _ in range(max(0, args.warmup)):
                mdl['req'].infer()
            cache[key] = mdl
        return cache[key]

    det_device, depth_device = args.yolo_device, args.device
    print('Keys: C/G/N detection device | Shift+C/G/N depth device | SPACE resolution')
    det_model = activate('detect', det_xml, det_device) if det_xml else None
    depth_model = activate('depth', depth_xml, depth_device) if depth_xml else None
    depth_on = depth_model is not None
    detect_on = det_model is not None
    # Names: default COCO labels from Ultralytics, not from the exported model.
    from ultralytics.utils import YAML
    names = {}
    if det_xml:
        metadata = det_xml.parent / 'metadata.yaml'
        if metadata.exists():
            names = {int(k): v for k, v in YAML.load(metadata).get('names', {}).items()}

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera')
    resolutions = [(args.width, args.height)] + [r for r in RESOLUTIONS if r != (args.width, args.height)]
    resolution_index = 0
    hue, saturation, contrast, brightness = 0.0, 1.0, 1.0, 0.0
    states = {'Left': new_hand_state(), 'Right': new_hand_state()}
    last_t = start_t = time.monotonic()
    timestamp_ms = -1
    fps = 0.0
    frame_index = 0
    cached_depth = None
    cached_boxes = []

    try:
        with vision.HandLandmarker.create_from_options(hand_opts) as hands, \
             vision.FaceLandmarker.create_from_options(face_opts) as faces, \
             vision.PoseLandmarker.create_from_options(pose_opts) as poses:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.monotonic()
                timestamp_ms = max(timestamp_ms + 1, int((now - start_t) * 1000))
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_result = hands.detect_for_video(image, timestamp_ms)
                face_result = faces.detect_for_video(image, timestamp_ms)
                pose_result = poses.detect_for_video(image, timestamp_ms)

                seen = set()
                overlays = []
                for pts, categories in zip(hand_result.hand_landmarks, hand_result.handedness):
                    if not categories:
                        continue
                    hand = categories[0].category_name
                    if hand not in states or hand in seen:
                        continue
                    seen.add(hand)
                    state = states[hand]
                    thumb = np.array([pts[4].x, pts[4].y], np.float32)
                    index = np.array([pts[8].x, pts[8].y], np.float32)
                    wrist = np.array([pts[0].x, pts[0].y], np.float32)
                    middle = np.array([pts[9].x, pts[9].y], np.float32)
                    palm_size = max(float(np.linalg.norm(wrist - middle)), 0.025)
                    pinch_ratio = float(np.linalg.norm(thumb - index)) / palm_size
                    pinching = pinch_ratio < (args.pinch_off if state['active'] else args.pinch_on)
                    cursor = (thumb + index) * 0.5
                    if pinching and not state['active']:
                        state['active'] = True
                        state['anchor'] = (float(cursor[0]), float(cursor[1]))
                        state['values'] = ((hue, saturation) if hand == 'Left' else (contrast, brightness))
                    elif not pinching:
                        state['active'] = False
                    if state['active']:
                        ax, ay = state['anchor']
                        v0, v1 = state['values']
                        dx, dy = float(cursor[0]) - ax, ay - float(cursor[1])
                        if hand == 'Left':
                            hue = float(np.clip(v0 + dx * 180, -90, 90))
                            saturation = float(np.clip(v1 + dy * 2.5, 0, 3))
                        else:
                            contrast = float(np.clip(v0 + dx * 2, 0.2, 3))
                            brightness = float(np.clip(v1 + dy * 180, -100, 100))
                    overlays.append((hand, points_px(pts, w, h), state['active'], cursor))
                for hand in states:
                    if hand not in seen:
                        states[hand]['active'] = False

                # Show the adjusted color image in the left window.
                # Keep output independent for depth and AI annotations.
                adjusted = apply_adjustments(frame, hue, saturation, contrast, brightness)
                output = adjusted.copy()
                if depth_on and frame_index % max(1, args.depth_every) == 0:
                    ov_preprocess(frame, depth_model)
                    depth_model['req'].infer()
                    depth_map = ov_depth_to_map(depth_model, w, h)
                    cached_depth = depth_to_color(depth_map, (w, h))
                if depth_on and cached_depth is not None:
                    output = cv2.addWeighted(output, 0.38, cached_depth, 0.62, 0)
                if detect_on and frame_index % max(1, args.detect_every) == 0:
                    ov_preprocess(frame, det_model)
                    det_model['req'].infer()
                    # Output is decoded in input-size coordinates then mapped to camera.
                    dets = decode_detections(det_model['output'][0], frame.shape,
                                             args.yolo_conf, input_size=(det_model['shape'][3], det_model['shape'][2]))
                    cached_boxes = []
                    for x1, y1, x2, y2, conf, cls in dets:
                        label = f'{names.get(int(cls), str(int(cls)))} {conf:.2f}'
                        cached_boxes.append(((int(x1), int(y1), int(x2), int(y2)), label))
                if detect_on:
                    for (x1, y1, x2, y2), label in cached_boxes:
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 240, 255), 2)
                        cv2.putText(output, label, (x1, max(12, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)

                for landmarks in pose_result.pose_landmarks:
                    draw_edges(output, points_px(landmarks, w, h), POSE_EDGES, (255, 80, 230), 3)
                # Sparse face mesh to avoid covering the depth map with hundreds of points.
                for landmarks in face_result.face_landmarks:
                    pts = points_px(landmarks, w, h)
                    for idx in range(0, len(pts), 6):
                        cv2.circle(output, pts[idx], 1, (240, 220, 80), -1)
                    for idx in (1, 4, 33, 61, 133, 152, 168, 263, 291, 362):
                        if idx < len(pts):
                            cv2.circle(output, pts[idx], 3, (0, 255, 255), -1)
                for hand, pts, active, cursor in overlays:
                    color = (80, 240, 80) if hand == 'Left' else (255, 180, 60)
                    draw_edges(output, pts, HAND_EDGES, color, 2)
                    for idx in (4, 8, 12, 16, 20):
                        cv2.circle(output, pts[idx], 5, color, -1)
                    cv2.line(output, pts[4], pts[8], color, 2)
                    cx, cy = int(cursor[0] * w), int(cursor[1] * h)
                    cv2.circle(output, (cx, cy), 13, color, 3 if active else 1)
                    cv2.putText(output, hand + (' GRAB' if active else ''),
                                (max(0, min(w - 130, cx + 15)), max(20, cy - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                dt = max(time.monotonic() - last_t, 1e-6)
                fps = 0.85 * fps + 0.15 / dt if fps else 1 / dt
                last_t = time.monotonic()
                draw_hud(output, (hue, saturation, contrast, brightness), states,
                         fps, (w, h), depth_on, detect_on)
                cv2.imshow('01 Adjusted Color', adjusted)
                cv2.imshow('02 Output - Depth + Objects + Landmarks', output)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    hue, saturation, contrast, brightness = 0.0, 1.0, 1.0, 0.0
                    states = {'Left': new_hand_state(), 'Right': new_hand_state()}
                elif key == ord('s'):
                    filename = time.strftime('hand_demo_009_%Y%m%d_%H%M%S.png')
                    cv2.imwrite(filename, output)
                    print('Saved:', filename)
                elif key == ord('d') and depth is not None:
                    depth_on = not depth_on
                elif key == ord('o') and yolo is not None:
                    detect_on = not detect_on
                elif key in (ord('c'), ord('g'), ord('n')) and det_xml:
                    target = {ord('c'): 'CPU', ord('g'): 'GPU', ord('n'): 'NPU'}[key]
                    try:
                        new_model = activate('detect', det_xml, target)
                        det_model, det_device = new_model, target
                        cached_boxes = []
                        print('Detection device:', det_device)
                    except Exception as exc:
                        print('Cannot change detection device:', exc)
                elif key in (ord('C'), ord('G'), ord('N')) and depth_xml:
                    target = {ord('C'): 'CPU', ord('G'): 'GPU', ord('N'): 'NPU'}[key]
                    try:
                        new_model = activate('depth', depth_xml, target)
                        depth_model, depth_device = new_model, target
                        cached_depth = None
                        print('Depth device:', depth_device)
                    except Exception as exc:
                        print('Cannot change depth device:', exc)
                elif key == ord(' '):
                    resolution_index = (resolution_index + 1) % len(resolutions)
                    rw, rh = resolutions[resolution_index]
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
                    cached_depth = None
                    cached_boxes = []
                    states = {'Left': new_hand_state(), 'Right': new_hand_state()}
                    print('Requested resolution:', rw, rh)
                frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
