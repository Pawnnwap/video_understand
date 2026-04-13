"""ocr_worker.py — standalone OCR subprocess
Called by vlm_analyser.run_ocr() to avoid PyTorch/PaddlePaddle CUDA conflict.

Usage (internal):
    python ocr_worker.py <frame_path> <lang> <min_confidence>

Outputs JSON to stdout: {"lines": [...], "error": null}
"""

import json
import os
import sys

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def main():
    if len(sys.argv) < 4:
        print(json.dumps({"lines": [], "error": "usage: ocr_worker.py <path> <lang> <conf>"}))
        sys.exit(1)

    frame_path = sys.argv[1]
    lang = sys.argv[2]
    min_conf = float(sys.argv[3])

    try:
        from paddleocr import PaddleOCR

        # Try each kwargs variant; stop at the first that succeeds
        ocr = None
        for kwargs in ({"use_gpu": True}, {"use_gpu": False}, {}):
            try:
                ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    show_log=False,
                    text_rec_score_thresh=min_conf,
                    **kwargs,
                )
                break
            except TypeError:
                continue
        if ocr is None:
            raise RuntimeError("PaddleOCR could not be initialised with any kwargs combination")
        # Use predict() (PP-OCRv5) if available, else ocr() (older)
        try:
            raw = ocr.predict(frame_path)
            # predict() returns list of dicts with 'rec_text'/'rec_score'
            lines = []
            if raw:
                for page in raw:
                    if not page:
                        continue
                    for item in (page if isinstance(page, list) else [page]):
                        if isinstance(item, dict):
                            text = item.get("rec_text", "")
                            conf = item.get("rec_score", 1.0)
                            if conf >= min_conf and text.strip():
                                lines.append(text.strip())
        except (AttributeError, TypeError):
            # Fallback to old ocr() API
            result = ocr.ocr(frame_path, cls=True)
            lines = []
            if result:
                for page in result:
                    if not page:
                        continue
                    for item in page:
                        if isinstance(item, dict):
                            text, conf = item.get("rec_text", ""), item.get("rec_score", 1.0)
                        else:
                            text, conf = item[1][0], item[1][1]
                        if conf >= min_conf and text.strip():
                            lines.append(text.strip())
        print(json.dumps({"lines": lines, "error": None}))
    except Exception as e:
        print(json.dumps({"lines": [], "error": str(e)}))


if __name__ == "__main__":
    main()
