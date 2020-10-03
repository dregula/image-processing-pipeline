import numpy as np
import cv2

from pipeline.pipeline import Pipeline
from pipeline.libs.template_detector import TemplateDetector


class DetectTemplateMatch(Pipeline):
    """
    Pipeline task to match a template image to a base (larger) image
        @summary:


    """

    def __init__(self, base_image: np.ndarray, cv_template_method, threshold, batch_size=1):
        self.base_image = base_image
        # note: the single bas-image is injected into the TemplateDetector class at initialization
        self.detector = TemplateDetector(base_image, cv_template_method=cv_template_method, threshold=threshold)
        self.batch_size = batch_size

        super(DetectTemplateMatch, self).__init__()

    def generator(self):
        """
        Yields the base-image enriched with matched template metadata
            @summary:
        """

        batch = []
        stop = False
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream
                data = next(self.source)
                batch.append(data)
            except StopIteration:
                stop = True

            # Check if there is anything in batch.
            # Process it if the size match batch_size or there is the end of the input stream.
            if len(batch) and (len(batch) == self.batch_size or stop):
                template_images = [data["template_image"] for data in batch]
                # Detect matches for all templates at once
                matches = self.detector.detect(self.base_image, template_images)

                # Extract the faces metadata and attache them to the proper image
                for template_image_idx, matches in matches.items():
                    batch[template_image_idx]["matches"] = matches

                # Yield all the data from buffer
                for data in batch:
                    if self.filter(data):
                        yield self.map(data)

                batch = []
