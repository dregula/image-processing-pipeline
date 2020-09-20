from pipeline.pipeline import Pipeline


class DisplayMatchSummary(Pipeline):
    def __init__(self):
        super(DisplayMatchSummary, self).__init__()

    def map(self, data):
        template_image_id = data["template_image_id"]
        match_rects = data["matches"]

        print(f"[INFO] {template_image_id}: template matches {len(match_rects)}")

        return data
