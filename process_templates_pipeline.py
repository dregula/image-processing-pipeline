import os
from tqdm import tqdm
import cv2

import pipeline.libs.utils as utils
from pipeline.capture_base_images import CaptureBaseImages
from pipeline.capture_template_images import CaptureTemplateImages
from pipeline.detect_template_match import DetectTemplateMatch
from pipeline.save_matches_as_marked_images import SaveMatches
from pipeline.save_match_summary import SaveMatchSummary
from pipeline.display_match_summary import DisplayMatchSummary
from pipeline.annotate_marked_image import AnnotateMarkedImage
from pipeline.display_marked_images import DisplayMarkedImages
# from pipeline.capture_video import CaptureVideo

def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Image processing pipeline")
    ap.add_argument("-b", "--base_dir", required=True,
                    help="path to base images directory")
    ap.add_argument("-t", "--template_dir", required=True,
                    help="path to template images directory")
    ap.add_argument("-o", "--output_dir", default="output",
                    help="path to output directory")
    ap.add_argument("-os", "--out-summary", default="summary.json",
                    help="output JSON summary file name")
    ap.add_argument("-c", "--cv_template_method", type=int, default=int(cv2.TM_CCOEFF_NORMED),
                    help="opencv match-template method")
    ap.add_argument("-th", "--threshold", type=float,  default=0.8,
                    help="threshold to filter weak template matches")
    ap.add_argument("-ov", "--out-video", default=None,
                    help="output video file name")
    ap.add_argument("-p", "--progress", action="store_true", help="display progress")
    ap.add_argument("-d", "--display", action="store_true", help="display video result")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch size")

    return ap.parse_args()


def main(args):
    # Create pipeline steps
    #  capture_video = CaptureVideo(int(args.input) if args.input.isdigit() else args.input)
    # capture_video = None

    capture_base_images = CaptureBaseImages(args.base_dir, level=0)
    # note: this stage takes the longest! consider parallelizing
    # TODO either abandon multiple base_images or refactor entirely for a single base_image
    if capture_base_images.has_next():
        test = capture_base_images.source
        list_cap_base = list(capture_base_images)
        first_capture_base_item = list_cap_base[0]
        if "base_image" in first_capture_base_item:
            base_image = first_capture_base_item["base_image"]

    capture_template_images = CaptureTemplateImages(args.template_dir, level=0)

    detect_template_matches = DetectTemplateMatch(base_image, cv_template_method=args.cv_template_method,
                                                  threshold=args.threshold, batch_size=args.batch_size)

    # TODO: messy to have to inject the base_image again!
    save_matches = SaveMatches(base_image, args.output_dir)
    summary_file = os.path.join(args.output_dir, args.out_summary)
    save_match_summary = SaveMatchSummary(summary_file)

    display_match_summary = DisplayMatchSummary()

    # annotate_marked_images = AnnotateMarkedImage("annotated_image") \
    #     if args.display or args.out_video else None

    # display_marked_images = DisplayMarkedImages("annotated_image") \
    #     if args.display else None

    # Create image processing pipeline
    pipeline = (capture_template_images |
                detect_template_matches |
                save_matches |
                save_match_summary |
                display_match_summary
                # annotate_marked_images |
                # display_marked_images
                )

# Iterate through pipeline
# if capture_video and capture_video is not None:
#     progress = tqdm(total=capture_video.frame_count if capture_video.frame_count > 0 else None,
#                     disable=not args.progress)
    try:
        for _ in pipeline:
            pass
            # progress.update(1)
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        pass
        # progress.close()

        # Pipeline cleanup
        # if capture_video:
        #     capture_video.cleanup()
        # if display_video:
        #     display_video.cleanup()
        # if save_video:
        #     save_video.cleanup()

    print(f"[INFO] Saving summary to {summary_file}...")
    save_match_summary.write()

    exit("done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
