# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from ultralytics.utils import LOGGER, YAML

AXERA_CHIPS = frozenset({"AX650", "AX8860"})
AXERA_PRECISIONS = {"AX650": "w8a8", "AX8860": "w16a16"}  # AX650 runs INT8, AX8860 runs 16-bit float
CALIBRATION_MAX_SIZE = 64  # cap staged images so a full COCO split cannot blow up the artifact and build time


def _calibration_images(dataset: Any) -> list[Path]:
    """Return original image paths from a calibration dataloader's dataset."""
    source = getattr(dataset, "dataset", dataset)
    image_paths = getattr(source, "im_files", None)
    if image_paths is None and hasattr(source, "samples"):
        image_paths = [x[0] for x in source.samples]
    if not image_paths:
        raise ValueError("AXERA export requires a calibration dataset with image file paths.")
    return [Path(x) for x in image_paths]


def _stage_calibration(images: list[Path], calib_dir: Path, calib_zip: Path) -> int:
    """Copy calibration images under space-free names and zip them for Pulsar2 Image mode."""
    shutil.rmtree(calib_dir, ignore_errors=True)
    calib_dir.mkdir(parents=True)
    for i, image in enumerate(images):
        suffix = image.suffix.lower() or ".jpg"
        shutil.copy2(image, calib_dir / f"{i}{suffix}")

    calib_zip.unlink(missing_ok=True)
    with zipfile.ZipFile(calib_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for image in sorted(calib_dir.iterdir()):
            if image.is_file():
                z.write(image, image.name)
    return len(images)


_PASSIVE_OPS = frozenset({"Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Cast", "Identity"})


def _graph_outputs(onnx_file: Path) -> tuple[list[str], list[str]]:
    """Return the ONNX graph output names and the layers Pulsar2 must keep in FP32.

    The primary YOLO output is a ``Concat`` that merges tensors with incompatible ranges: box coordinates (~0-640) with
    class scores, keypoint confidences or OBB angles (~0-1). A single U8 scale flattens the small-range tensors to zero,
    so every op that produces a branch of that concat stays FP32. The walk descends through pass-through ops and nested
    concats but stops at the first real compute op, which keeps the set small and avoids nodes that Pulsar2 fuses.
    """
    import onnx

    graph = onnx.load(str(onnx_file), load_external_data=False).graph
    producers = {output: node for node in graph.node for output in node.output}
    outputs = [output.name for output in graph.output]
    if not outputs:
        return outputs, []

    primary = producers.get(outputs[0])
    if primary is None or primary.op_type != "Concat":
        # Heads without a mixed-range concat (classification) still need FP32 softmaxes to keep probability resolution.
        return outputs, [node.name for node in graph.node if node.op_type == "Softmax"]

    layers, stack, seen = {primary.name}, [primary], {primary.name}
    while stack:
        node = stack.pop()
        for tensor in node.input:
            upstream = producers.get(tensor)
            if upstream is None or upstream.name in seen:
                continue
            seen.add(upstream.name)
            layers.add(upstream.name)
            if upstream.op_type in _PASSIVE_OPS or upstream.op_type == "Concat":
                stack.append(upstream)
    return outputs, sorted(layers)


def _build_config(
    onnx_file: Path,
    build_dir: Path,
    calib_zip: Path,
    calibration_size: int,
    target: str,
    precision: str,
    output_layers: list[str],
) -> dict:
    """Build the Pulsar2 configuration for the precision the target chip supports."""
    if precision == "w16a16":
        # AX8860 runs 16-bit float. axnn 8.0 rejects `data_type: FP16` (FP16 is absent from the quantizer dtype map and
        # only BF16 is accepted), so the 16-bit float build uses BF16 for both weights and activations.
        layer_configs = [
            {
                "start_tensor_names": ["DEFAULT"],
                "end_tensor_names": ["DEFAULT"],
                "data_type": "BF16",
                "weight_data_type": "BF16",
            }
        ]
    else:  # AX650: INT8 weights and activations, with the mixed-range head branches kept in FP32
        layer_configs = [
            {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "U8"},
            *({"layer_name": name, "data_type": "FP32", "output_data_type": "FP32"} for name in output_layers),
        ]
    return {
        "input": str(onnx_file),
        "output_dir": str(build_dir),
        "output_name": "compiled.axmodel",
        "model_type": "ONNX",
        "target_hardware": target,
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "DEFAULT",
                    "calibration_dataset": str(calib_zip),
                    "calibration_format": "Image",
                    "calibration_size": calibration_size,
                    "calibration_mean": [0, 0, 0],
                    "calibration_std": [255, 255, 255],
                }
            ],
            "layer_configs": layer_configs,
            "calibration_method": "MinMax",
        },
        "input_processors": [
            {
                "tensor_name": "DEFAULT",
                "tensor_format": "RGB",
                "tensor_layout": "NCHW",
                "src_format": "BGR",
                "src_layout": "NHWC",
                "src_dtype": "U8",
                "mean": [0, 0, 0],
                "std": [255, 255, 255],
            }
        ],
        "output_processors": [{"tensor_name": "DEFAULT", "output_dtype": "FP32"}],
    }


def _api_build(config: dict, prefix: str) -> bool:
    """Compile with the in-process axnn build API, returning False when the API is unavailable."""
    try:
        import axnn.yamain.config.build_config_pb2 as build_pb
        from axnn.yamain.command.build import build
        from google.protobuf.json_format import ParseDict
    except (ImportError, ModuleNotFoundError) as e:
        LOGGER.warning(f"{prefix} axnn build API unavailable: {e}")
        return False

    cfg = build_pb.BuildConfig()
    ParseDict(config, cfg)
    LOGGER.info(f"{prefix} starting export with axnn build API...")
    build(cfg)
    return True


def _cli_build(config_file: Path, prefix: str) -> None:
    """Compile with the pulsar2 CLI as a fallback when the API is unavailable."""
    executable = shutil.which("pulsar2")
    if not executable:
        raise FileNotFoundError("AXERA export requires the axnn build API or the pulsar2 CLI.")
    LOGGER.warning(f"{prefix} axnn build API unavailable, falling back to pulsar2 build --config.")
    subprocess.run([executable, "build", "--config", str(config_file)], check=True)


def _check_build(build_dir: Path, prefix: str) -> dict:
    """Validate Pulsar2 output and return its build context."""
    build_context = build_dir / "build_context.json"
    if not build_context.is_file():
        raise RuntimeError(f"{prefix} Pulsar2 did not create build_context.json.")
    context = json.loads(build_context.read_text())
    if context.get("code") != "Success":
        raise RuntimeError(f"{prefix} Pulsar2 build failed: {context.get('error_msg', context.get('code'))}")
    if not (build_dir / "compiled.axmodel").is_file():
        raise RuntimeError(f"{prefix} Pulsar2 did not create compiled.axmodel.")
    return context


def onnx2axera(
    onnx_file: str | Path,
    output_dir: str | Path,
    dataset: Any,
    metadata: dict | None = None,
    target: str = "AX650",
    prefix: str = "",
) -> str:
    """Compile an ONNX model to an AXERA ``.axmodel`` directory with Pulsar2.

    Args:
        onnx_file (str | Path): Source ONNX model path.
        output_dir (str | Path): Directory for the final ``_axera_model`` artifact.
        dataset (Any): Calibration dataloader from ``Exporter.get_int8_calibration_dataloader``.
        metadata (dict | None): Ultralytics metadata saved beside the compiled model.
        target (str): Pulsar2 ``target_hardware`` name: ``"AX650"`` (W8A8) or ``"AX8860"`` (W16A16/FP16).
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_axera_model`` directory.
    """
    onnx_file = Path(onnx_file).resolve()
    output_dir = Path(output_dir).resolve()
    target = str(target).upper()
    if target not in AXERA_CHIPS:
        raise ValueError(f"Invalid AXERA target '{target}'. Valid targets are {sorted(AXERA_CHIPS)}.")

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    target_onnx = output_dir / "model.onnx"
    if onnx_file != target_onnx:
        shutil.move(str(onnx_file), target_onnx)
    onnx_file = target_onnx
    build_dir = output_dir

    images = _calibration_images(dataset)
    if len(images) > CALIBRATION_MAX_SIZE:
        LOGGER.info(
            f"{prefix} using the first {CALIBRATION_MAX_SIZE} of {len(images)} calibration images "
            f"(set fraction=<n> on export to choose a different subset)"
        )
        images = images[:CALIBRATION_MAX_SIZE]
    calib_dir = build_dir / "calib"
    calib_zip = build_dir / "calib.zip"
    calibration_size = _stage_calibration(images, calib_dir, calib_zip)

    output_names, output_layers = _graph_outputs(onnx_file)
    precision = AXERA_PRECISIONS[target]

    config = _build_config(onnx_file, build_dir, calib_zip, calibration_size, target, precision, output_layers)
    config_file = output_dir / "config.json"
    config_file.write_text(json.dumps(config, indent=2) + "\n")

    if not _api_build(config, prefix):
        _cli_build(config_file, prefix)

    context = _check_build(build_dir, prefix)

    if metadata is not None:
        meta = dict(metadata)
        meta["dynamic"] = False
        meta["axera"] = {
            "target": target,
            "inputs": [{"name": "images", "dtype": "uint8", "layout": "NHWC", "range": [0, 255]}],
            "outputs": [{"name": n, "dtype": "float32", "order": i} for i, n in enumerate(output_names)],
            "precision": precision,
            "calibration": {"format": "Image", "size": calibration_size},
            "build_context": context.get("code", ""),
        }
        YAML.save(output_dir / "metadata.yaml", meta)

    return str(output_dir)
