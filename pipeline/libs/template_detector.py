import cv2
import numpy as np


class TemplateDetector:
    def __init__(self, cv_template_method, threshold):
        # one of the cv2.TM_...NORMED constants
        self.cv_template_method = cv_template_method
        self.threshold = threshold

    def detect(self, base_images, template_images):

        # Prepare storage for matches for every template_image in the batch
        matches = dict(zip(range(len(template_images)), [[] for _ in range(len(template_images))]))

        # loop over the templates
        #  DEBUG:
        for template_image_idx, template_image in enumerate(template_images):
            # TODO: revise to accomodate multiple base images
            #   for now: just submit the first of the list...
            result = cv2.matchTemplate(base_images[0], template_image, self.cv_template_method)
            # TODO: extract filtering to another pipeline element
            # TODO: or an injected lib
            filtered = np.where(result >= self.threshold)

            confidence_at_location = result[filtered]
            matched_points = list(zip(*filtered[::-1]))
            matched_points_with_value = list(zip(matched_points, confidence_at_location))

            w, h, c = template_image.shape

            # append each (box, value) tuple in filtered_results
            box = [0, 0, 0, 0]
            for pt, confidence in matched_points_with_value:

                # box: (x1, x2, y1, y2); recall maxLoc point is h, w
                box = pt[0], pt[1], pt[0] + h, pt[1] + w
                # Add result
                matches[template_image_idx].append((box, confidence))

        return matches
