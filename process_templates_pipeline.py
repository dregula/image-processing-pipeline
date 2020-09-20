import os
import cv2

import pipeline.libs.utils as utils
from pipeline.capture_base_images import CaptureBaseImages
from pipeline.capture_template_images import CaptureTemplateImages
from pipeline.detect_template_match import DetectTemplateMatch
from pipeline.save_matches_as_marked_images import SaveMatches
from pipeline.save_match_summary import SaveMatchSummary
from pipeline.display_match_summary import DisplayMatchSummary


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
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch size")

    return ap.parse_args()


def main(args):
    # Create pipeline steps

    capture_base_images = CaptureBaseImages(args.base_dir, level=1)

    capture_template_images = CaptureTemplateImages(args.template_dir, level=1)

    detect_template_matches = DetectTemplateMatch(cv_template_method=args.cv_template_method,
                                                  threshold=args.threshold, batch_size=args.batch_size)

    save_matches = SaveMatches(args.output_dir)
    summary_file = os.path.join(args.output_dir, args.out_summary)
    save_match_summary = SaveMatchSummary(summary_file)

    display_match_summary = DisplayMatchSummary()

    # Create image processing pipeline
    pipeline = (capture_base_images |
                capture_template_images |
                detect_template_matches |
                save_matches |
                save_match_summary |
                display_match_summary)

    try:
        # Iterate through pipeline
        for _ in pipeline:
            pass
    except StopIteration:
        return
    except KeyboardInterrupt:
        return
    finally:
        print(f"[INFO] Saving summary to {summary_file}...")
        save_match_summary.write()

    exit("done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
