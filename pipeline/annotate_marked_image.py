import cv2

from pipeline.pipeline import Pipeline
from pipeline.libs.colors import colors
from pipeline.libs.text import put_text


class AnnotateMarkedImage(Pipeline):
    """Pipeline task for image annotation."""

    def __init__(self, dst):
        self.dst = dst
        super(AnnotateMarkedImage, self).__init__()

    def map(self, data):
        data = self.annotate_matches(data)

        return data

    def annotate_matches(self, data):
        """Annotates matches on the image with bounding box and confidence info."""

        if "matches" not in data:
            return data

        annotated_image = data["base_image"].copy()
        matches = data["matches"]

        # Loop over the matches and draw a rectangle around each
        for i, face in enumerate(matches):
            box, confidence = face
            (x1, y1, x2, y2) = np.array(box).astype("int")
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), colors.get("green").to_bgr(), 2)
            put_text(annotated_image, f"{confidence:.2f}", (x1 - 1, y1),
                     color=colors.get("white").to_bgr(),
                     bg_color=colors.get("green").to_bgr(),
                     org_pos="bl")

        data[self.dst] = annotated_image

        return data
