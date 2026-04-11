"""Long-running YOLO inference worker. Communicates via JSON lines over stdin/stdout."""
import sys
import json

_models = {}

def _get_model(name, model_path):
    if name not in _models:
        from ultralytics import YOLO
        _models[name] = YOLO(model_path)
    return _models[name]

def _predict(model_name, model_path, image_path, conf=0.5):
    model = _get_model(model_name, model_path)
    results = model(image_path, conf=conf, verbose=False)
    detections = []
    for r in results:
        ih, iw = r.orig_shape
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2) / iw
            cy = ((y1 + y2) / 2) / ih
            w = (x2 - x1) / iw
            h = (y2 - y1) / ih
            detections.append({
                'cls': int(box.cls[0].item()),
                'cx': round(cx, 6), 'cy': round(cy, 6),
                'w': round(w, 6), 'h': round(h, 6),
                'confidence': round(box.conf[0].item(), 4),
            })
    return detections

def main():
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            req_id = cmd.get('id', 0)
            action = cmd.get('action')
            if action == 'ping':
                print(json.dumps({'id': req_id, 'result': 'pong'}), flush=True)
            elif action == 'predict':
                dets = _predict(cmd['model'], cmd['modelPath'], cmd['imagePath'], cmd.get('conf', 0.5))
                print(json.dumps({'id': req_id, 'result': dets}), flush=True)
            else:
                print(json.dumps({'id': req_id, 'error': f'Unknown action: {action}'}), flush=True)
        except Exception as e:
            print(json.dumps({'id': 0, 'error': str(e)}), flush=True)

if __name__ == '__main__':
    main()
