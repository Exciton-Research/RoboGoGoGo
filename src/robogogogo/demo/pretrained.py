"""Local pretrained text-conditioned detection. No catalog references or object metadata."""

import base64
import hashlib
import importlib.util
import io
import json
import re
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"


class DetectionRejected(ValueError):
    """A valid text query was not confidently grounded in the image."""


class ModelNotReady(RuntimeError):
    pass


def model_directory(cache):
    return Path(cache) / "models" / "owlv2"


def model_ready(cache):
    folder = model_directory(cache)
    return (
        (folder / "source.json").is_file()
        and (folder / "model.safetensors").is_file()
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def download_model(cache):
    """Explicit one-time setup; runtime inference never downloads files."""
    from huggingface_hub import snapshot_download

    folder = model_directory(cache)
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=folder,
        allow_patterns=["*.json", "merges.txt", "README.md", "model.safetensors"],
    )
    with (folder / "model.safetensors").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (folder / "source.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "sha256": digest,
                "license": "Apache-2.0",
                "source": f"https://huggingface.co/{MODEL_ID}",
            },
            indent=2,
        )
        + "\n"
    )
    return folder


def box_iou(first, second):
    first, second = np.asarray(first), np.asarray(second)
    intersection = np.maximum(
        0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])
    ).prod()
    union = (
        np.maximum(0, first[2:] - first[:2]).prod()
        + np.maximum(0, second[2:] - second[:2]).prod()
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0


class PretrainedDetector:
    def __init__(self, cache, *, device=None):
        if not model_ready(cache):
            raise ModelNotReady(
                "OWLv2 is not installed. Install '.[vlm]' and run "
                "robogogogo-demo --setup-vlm, then restart the demo."
            )
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        folder = model_directory(cache)
        source = json.loads((folder / "source.json").read_text())
        with (folder / "model.safetensors").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if (source.get("model"), source.get("revision"), source.get("sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            digest,
        ):
            raise ModelNotReady("OWLv2 cache changed; rerun robogogogo-demo --setup-vlm")
        self._image_cache = OrderedDict()
        self.torch = torch
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.processor = Owlv2Processor.from_pretrained(
            folder, local_files_only=True, use_fast=False
        )
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(
                folder,
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
            )
            .eval()
            .to(self.device)
        )

    def detect(self, rgb, query, *, threshold=0.2):
        """Text + RGB in, scored pixel boxes out. Works independently of Panda assets."""
        query = " ".join(query.split())
        if not query or len(query) > 120:
            raise ValueError("Describe the object using 1–120 characters")
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("The detector requires a uint8 RGB image")
        image = Image.fromarray(rgb)
        text = [[f"a photo of {query}"]]
        inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        # Cache visual features by exact RGB bytes, not scene IDs or selected
        # targets. A fresh sensor frame always determines whether reuse is valid.
        key = (rgb.shape, hashlib.sha256(rgb.tobytes()).digest())
        with self.torch.inference_mode():
            if key not in self._image_cache:
                feature_map, _ = self.model.image_embedder(inputs["pixel_values"])
                features = feature_map.flatten(1, 2)
                boxes = self.model.box_predictor(features, feature_map)
                self._image_cache[key] = (features, boxes)
                if len(self._image_cache) > 48:
                    self._image_cache.popitem(last=False)
            self._image_cache.move_to_end(key)
            features, boxes = self._image_cache[key]
            text_features = self.model.owlv2.get_text_features(
                inputs["input_ids"], inputs["attention_mask"]
            ).unsqueeze(0)
            text_features = text_features / self.torch.linalg.norm(
                text_features, dim=-1, keepdim=True
            )
            logits, _ = self.model.class_predictor(
                features, text_features, inputs["input_ids"][:, 0].unsqueeze(0) > 0
            )
            output = SimpleNamespace(logits=logits, pred_boxes=boxes)
        result = self.processor.post_process_grounded_object_detection(
            output,
            threshold=threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=text,
        )[0]
        detections = []
        for score, box in sorted(
            zip(result["scores"].cpu().tolist(), result["boxes"].cpu().tolist()),
            reverse=True,
        ):
            box = np.clip(box, [0, 0, 0, 0], [image.width, image.height] * 2).tolist()
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if all(box_iou(box, item["box"]) < 0.45 for item in detections):
                detections.append({"box": box, "score": score, "query": query})
        return detections

    def find_target(self, query, capture):
        """Inspect fixed RGB-D views; the sensor callback supplies no object state."""
        from .vision import workspace_crop

        started = time.perf_counter()
        for view, focused in (("side", False), ("front", True)):
            sensor = capture(view=view)
            try:
                crop = workspace_crop(**sensor) if focused else None
                result = self.find(query, **sensor, crop=crop)
                if result["position"] is None:
                    raise DetectionRejected(result["pickup_error"])
                result["camera_view"] = view
                if crop is not None:
                    result["detection_crop"] = crop
                result["inference_seconds"] = time.perf_counter() - started
                return result
            except DetectionRejected:
                pass
        confirmed = []
        for view in ("side", "diagonal", "front", "left"):
            try:
                result = self.inspect(query, **capture(view=view, detail=True))
                if result["position"] is None:
                    raise DetectionRejected(result["pickup_error"])
                result["camera_view"] = view
                for other in confirmed:
                    distance = np.linalg.norm(np.asarray(result["position"]) - other["position"])
                    if distance < 0.035:
                        best = max((result, other), key=lambda item: item["score"])
                        best.update(
                            inspection_views=[other["camera_view"], view],
                            inference_seconds=time.perf_counter() - started,
                        )
                        return best
                confirmed.append(result)
            except DetectionRejected:
                pass
        raise DetectionRejected(
            f"OWLv2 could not confirm '{query}' from two inspection views. No motion started."
        )

    def inspect(self, query, *, rgb, depth, position, rotation, fovy):
        """Compare close-ups of all visible depth regions using the user's words."""
        from .vision import proposals

        started = time.perf_counter()
        query = " ".join(query.split())
        if not query or len(query) > 120:
            raise ValueError("Describe the object using 1–120 characters")
        # Grammatical container phrasing, independent of any library: e.g.
        # "potted meat can" -> "a can of potted meat". No brand/asset aliases.
        match = re.fullmatch(r"(.+) (box|carton|can|tin)", query, re.IGNORECASE)
        prompt = f"a {match[2]} of {match[1]}" if match else query
        candidates = []
        for region in proposals(rgb, depth, position, rotation, fovy, with_appearance=False):
            left, top, right, bottom = region["box"]
            pad = max(12, int(max(right - left, bottom - top) * 0.15))
            left, top = max(0, left - pad), max(0, top - pad)
            right, bottom = min(rgb.shape[1], right + pad), min(rgb.shape[0], bottom + pad)
            detections = self.detect(rgb[top:bottom, left:right], prompt, threshold=0.05)
            for detection in detections:
                box = (np.asarray(detection["box"]) + [left, top, left, top]).tolist()
                # Recognition must cover the physical object, not a small photo
                # or printed word on its packaging. Keep every competing region.
                if box_iou(box, region["box"]) >= 0.35:
                    candidates.append({**detection, "box": box, "query": query})
                    break
        candidates.sort(key=lambda item: item["score"], reverse=True)
        result = self._localize(
            query, candidates, rgb=rgb, depth=depth, position=position, rotation=rotation, fovy=fovy
        )
        result.update(
            inspection="object close-ups",
            detector_prompt=prompt,
            inference_seconds=time.perf_counter() - started,
        )
        return result

    def find(self, query, *, rgb, depth, position, rotation, fovy, crop=None):
        """Locate one text detection with depth; never accepts catalog IDs or dimensions."""
        started = time.perf_counter()
        # Keep weak competitors for the ambiguity check, even if they fall just
        # below the acceptance threshold. Filtering them first hides near ties.
        if crop is None:
            detections = self.detect(rgb, query, threshold=0.05)
        else:
            left, top, right, bottom = crop
            if not (0 <= left < right <= rgb.shape[1] and 0 <= top < bottom <= rgb.shape[0]):
                raise ValueError("Detection crop must lie within the camera image")
            detections = self.detect(rgb[top:bottom, left:right], query, threshold=0.05)
            detections = [
                {**item, "box": (np.asarray(item["box"]) + [left, top, left, top]).tolist()}
                for item in detections
            ]
        result = self._localize(
            query, detections, rgb=rgb, depth=depth, position=position, rotation=rotation, fovy=fovy
        )
        result["inference_seconds"] = time.perf_counter() - started
        return result

    def _localize(self, query, detections, *, rgb, depth, position, rotation, fovy):
        from .vision import localize_box

        if not detections or detections[0]["score"] < 0.2:
            raise DetectionRejected(
                f"OWLv2 did not confidently detect '{query}'. No motion started."
            )
        best = detections[0]
        query = best["query"]
        margin = best["score"] - detections[1]["score"] if len(detections) > 1 else best["score"]
        if len(detections) > 1 and margin < 0.06:
            raise DetectionRejected(
                "OWLv2 found ambiguous instances. Use a more specific description."
            )
        try:
            pick, region, grasp = localize_box(
                best["box"], rgb=rgb, depth=depth, position=position, rotation=rotation, fovy=fovy
            )
            pickup_error = None
        except ValueError as error:
            pick, region, grasp, pickup_error = None, None, None, str(error)
        annotated = Image.fromarray(rgb)
        draw = ImageDraw.Draw(annotated)
        draw.rectangle(best["box"], outline="#ffb340", width=3)
        draw.text((best["box"][0], max(0, best["box"][1] - 14)), query, fill="#ffb340")
        output = io.BytesIO()
        annotated.save(output, format="PNG")
        return {
            "name": query,
            "query": query,
            "position": None if pick is None else pick.tolist(),
            "grasp_position": None if grasp is None else grasp.tolist(),
            "pickup_error": pickup_error,
            "box": best["box"],
            "depth_box": region,
            "score": best["score"],
            "margin": margin,
            "image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(),
            "method": "Pretrained OWLv2 text detection + calibrated depth; no catalog references or dimensions",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "device": self.device,
        }
