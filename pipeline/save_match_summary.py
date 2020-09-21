import os
import json
import numpy as np

from pipeline.pipeline import Pipeline


class SaveMatchSummary(Pipeline):
    """Pipeline task to save processing summary."""

    def __init__(self, filename):
        self.filename = filename

        self.summary = {}
        super(SaveMatchSummary, self).__init__()

    def map(self, data):
        template_image_id = data["template_image_id"]
        match_files = data["match_files"]
        matches = data["matches"]

        # Loop over all detected faces and buffer summary results
        self.summary[template_image_id] = {}
        for i, match in enumerate(matches):
            box, confidence = match
            (x1, y1, x2, y2) = np.array(box).astype("int")
            match_file = match_files[i]
            self.summary[template_image_id][match_file] = {
                "box": np.array([x1, y1, x2, y2], dtype=int).tolist(),
                "confidence": confidence.item()
            }

        return data

    def write(self):
        dirname = os.path.dirname(os.path.abspath(self.filename))
        os.makedirs(dirname, exist_ok=True)

        with open(self.filename, 'w') as json_file:
            json_file.write(json.dumps(self.summary))
