import os
import cv2
import numpy as np

from pipeline.pipeline import Pipeline


class SaveMatches(Pipeline):
    """Pipeline task to save detected matches."""

    def __init__(self, base_image: np.ndarray, path, image_ext="jpg"):
        self.base_image = base_image
        self.path = path
        self.image_ext = image_ext

        super(SaveMatches, self).__init__()

    @staticmethod
    def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def map(self, data):
        template_image_id = data["template_image_id"]
        matches = data["matches"]
        data["match_files"] = []

        # Loop over all detected faces
        for i, match in enumerate(matches):
            box, confidence = match
            marked_image = self.base_image.copy()
            # do not allow a negative number,as the image, cannot be cropped with below
            (x1, y1, x2, y2) = np.maximum(np.array(box).astype("int"), 0)

            cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 255, 255), 16)

            # TODO: refer to SlideHosting project and asujtable window size
            resize = self.resize_image(marked_image, 1024, 960)

            # TODO: devise and place a title on the marked image
            # img_to_show = self.place_title(resize, match_template_item.title)

            output = os.path.join(*(template_image_id.split(os.path.sep)))
            output = os.path.join(self.path, output)
            # TODO: output is a new directory (tree) with exactly the template file name (including suffix)
            # e.g. 'output\\assets/images/templates\\HE_99E1_p3_cropped_TEMPLATE.tif\\00000.jpg'
            os.makedirs(output, exist_ok=True)

            # Save matches
            match_file = os.path.join(output, f"{i:05d}.{self.image_ext}")
            data["match_files"].append(match_file)
            try:
                cv2.imwrite(match_file, resize)
            except OSError as oe:
                print(f"[OSERROR] {oe}")
            except AssertionError as ae:
                print(f"[ASSERTIONERROR] {ae}")
            except BaseException as e:
                print(f"[ERROR] {e}")

        return data
