import numpy as np

# Using the tracking info returned by ultralytics YOLO.track (it includes .boxes with .id when track is used)
# This file provides helper functions to parse tracked results
def parse_frame_results(res, model):
    """
    Parses detection+tracking results from a single frame from the YOLO model.
    This function is optimized to avoid per-box CPU/Numpy conversions.
    """
    persons = []
    balls = []

    if res.boxes is None or not hasattr(res.boxes, 'id') or res.boxes.id is None:
        return persons, balls

    try:
        # Optimized path: get all data as numpy arrays in one go
        track_ids = res.boxes.id.int().cpu().numpy()
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.int().cpu().numpy()
        class_names = model.names

        for i in range(len(track_ids)):
            item = {
                'id': track_ids[i],
                'box': boxes_xyxy[i].tolist(),
                'conf': confs[i]
            }
            class_name = class_names.get(classes[i], str(classes[i]))

            if class_name == 'person':
                persons.append(item)
            elif class_name in ['sports ball', 'ball', 'sports_ball']:
                balls.append(item)

    except Exception as e:
        # Fallback path for safety
        print(f"Optimized parsing failed ({e}), falling back to slower method for this frame.")
        persons, balls = [], []
        for box in res.boxes:
            if not hasattr(box, 'id') or box.id is None or not hasattr(box, 'cls') or box.cls is None:
                continue

            try:
                name = model.names.get(int(box.cls.item()), str(int(box.cls.item())))
                track_id = int(box.id.item())
                item = {
                    'id': track_id,
                    'box': box.xyxy.squeeze().cpu().tolist(),
                    'conf': box.conf.item()
                }
                if name == 'person':
                    persons.append(item)
                elif name in ['sports ball', 'ball', 'sports_ball']:
                    balls.append(item)
            except Exception:
                continue

    return persons, balls