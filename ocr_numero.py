import easyocr
import cv2
import numpy as np

class JerseyOCR:
    def __init__(self, langs=['en']):
        print('Initialising EasyOCR reader (this may take a while)')
        self.reader = easyocr.Reader(lang_list=langs, gpu=False)

    def read_number(self, frame, box):
        # crop the central area of box, try OCR
        x1,y1,x2,y2 = map(int, box)
        h = y2-y1
        w = x2-x1
        # focus on upper back/torso area (if bbox is full body)
        cy1 = y1 + int(h*0.2)
        cy2 = y1 + int(h*0.8)
        crop = frame[cy1:cy2, x1:x2]
        if crop.size == 0:
            return None
        try:
            results = self.reader.readtext(crop)
            # find largest numeric result
            nums = []
            for bbox, txt, prob in results:
                s = ''.join([c for c in txt if c.isdigit()])
                if len(s)>0:
                    nums.append((s, prob))
            if nums:
                nums = sorted(nums, key=lambda x:-x[1])
                return nums[0][0]
        except Exception as e:
            return None
        return None
