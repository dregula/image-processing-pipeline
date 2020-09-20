import cv2

from pipeline.pipeline import Pipeline
import pipeline.libs.utils as utils


class CaptureTemplateImages(Pipeline):
    """Pipeline task to capture images from directory"""

    def __init__(self, src_dir, valid_exts=(".tiff", ".tif", ".jpg", ".png"), level=None, batch_size=1):
        self.src_dir = src_dir
        self.valid_exts = valid_exts
        self.level = level
        self.batch_size = batch_size

        super(CaptureTemplateImages, self).__init__()

    def generator(self):
        """Yields the image content and metadata."""

        batch = []
        stop = False

        template_source = utils.list_files(self.src_dir, self.valid_exts, self.level)
        while self.has_next() and not stop:
            try:
                # Buffer the pipeline stream
                data = next(self.source)
                batch.append(data)
            except StopIteration:
                stop = True

            if len(batch) and (len(batch) == self.batch_size or stop):
                # TODO: handle multiple base_images
                base_image_file = batch[0]["base_image_file"]
                base_image = batch[0]["base_image"]
                for template_image_idx, template_image_file in enumerate(template_source):
                    template_image = cv2.imread(template_image_file)
                    # promote the base_image
                    # TODO: this duplicates the base Image in EVERY template
                    batch[template_image_idx]["base_image_file"] = base_image_file
                    batch[template_image_idx]["base_image"] = base_image
                    batch[template_image_idx]["template_image_id"] = template_image_file
                    batch[template_image_idx]["template_image"] = template_image

                # Yield all the data from buffer
                for data in batch:
                    if self.filter(data):
                        debugYIELD = self.map(data)
                        yield self.map(data)

                batch = []
