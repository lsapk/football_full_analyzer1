# Using the tracking info returned by ultralytics YOLO.track (it includes .boxes with .id when track is used)
# This file provides helper functions to parse tracked results
def parse_frame_results(res, model):
    persons = []
    balls = []
    try:
        boxes = res.boxes
    except Exception:
        return persons, balls
    for box in boxes:
        try:
            cls = int(box.cls.cpu().numpy())
            conf = float(box.conf.cpu().numpy())
            xyxy = box.xyxy.cpu().numpy().tolist()[0]
        except Exception:
            # fallback if not tensor-wrapped
            try:
                cls = int(box.cls)
                conf = float(box.conf)
                xyxy = box.xyxy
            except Exception:
                continue
        name = model.model.names.get(cls, str(cls))
        track_id = None
        if hasattr(box, 'id'):
            try:
                track_id = int(box.id.cpu().numpy())
            except Exception:
                try:
                    track_id = int(box.id)
                except Exception:
                    track_id = None
        if name == 'person':
            persons.append({'id':track_id, 'box':xyxy, 'conf':conf})
        elif name in ['sports ball','ball','sports_ball']:
            balls.append({'id':track_id, 'box':xyxy, 'conf':conf})
    return persons, balls
