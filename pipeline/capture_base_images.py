import cv2

from pipeline.pipeline import Pipeline
import pipeline.libs.utils as utils

# TRY: 2020-09-20 ignore this class and use a single base_image in main()
class CaptureBaseImages(Pipeline):
    """Pipeline task to capture images from directory"""

    def __init__(self, src_dir, valid_exts=(".tiff", ".tif", ".jpg", ".png"), level=None):
        self.src_dir = src_dir
        self.valid_exts = valid_exts
        self.level = level

        super(CaptureBaseImages, self).__init__()

    def generator(self):
        """Yields the image content and metadata."""

        base_image_source = utils.list_files(self.src_dir, self.valid_exts, self.level)
        while self.has_next():
            try:
                # TODO: revise to use enumerate() instead of next() for multiple base_images
                base_image_file = next(base_image_source)
                base_image = cv2.imread(base_image_file)

                data = {
                    "base_image_file": base_image_file,
                    "base_image": base_image,
                }

                if self.filter(data):
                    debugYIELD = self.map(data)
                    yield self.map(data)
            except StopIteration:
                return
