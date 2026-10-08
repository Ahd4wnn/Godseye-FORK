"""TensorRT build/run helpers for NanoOWL's OWL-ViT image encoder.

NanoOWL's own `load_image_encoder_engine` depends on `torch2trt.TRTModule` and `trtexec`, both of which assume the
legacy TensorRT binding API that no longer exists in TensorRT 10+/11. This module replaces ONLY that loader: the ONNX
graph comes from NanoOWL's `OwlPredictor.export_image_encoder_onnx` and the engine output is wrapped into NanoOWL's own
`OwlEncodeImageOutput`, so `OwlPredictor.encode_image` / `decode` are used unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger("godseye.perception")

_OUTPUT_NAMES = ("image_embeds", "image_class_embeds", "logit_shift", "logit_scale", "pred_boxes")


class TrtError(RuntimeError):
    pass


def export_onnx(predictor, onnx_path: str | Path, half: bool, opset: int = 18) -> None:
    """Export NanoOWL's `encode_image_torch` (unchanged) to ONNX. TensorRT 11 networks are strongly typed, so FP16
    must be chosen here, by exporting a half-precision copy of the model."""
    import copy

    class Enc(torch.nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def forward(self, image):
            o = self.parent.encode_image_torch(image)
            return o.image_embeds, o.image_class_embeds, o.logit_shift, o.logit_scale, o.pred_boxes

    src = copy.deepcopy(predictor)
    if half:
        src.model = src.model.half()
        src.box_bias = src.box_bias.half()
    dtype = torch.float16 if half else torch.float32
    x = torch.randn(1, 3, src.image_size, src.image_size, device=src.device, dtype=dtype)
    axes = {n: {0: "batch"} for n in ("image",) + _OUTPUT_NAMES}
    torch.onnx.export(Enc(src).eval(), x, str(onnx_path), input_names=["image"], output_names=list(_OUTPUT_NAMES),
                      dynamic_axes=axes, opset_version=opset)


def build_engine(onnx_path: str | Path, engine_path: str | Path, image_size: int, max_batch: int = 1, fp16: bool = True) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise TrtError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    shape = lambda b: (b, 3, image_size, image_size)  # noqa: E731
    profile.set_shape("image", shape(1), shape(1), shape(max_batch))
    cfg.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        raise TrtError("TensorRT engine build failed")
    Path(engine_path).write_bytes(bytes(blob))


def _dims(d) -> tuple:
    # Never `tuple(d)`: TensorRT's Dims iteration ends via a C++ IndexError, and once GStreamer is loaded in the same
    # process libunwind hijacks _Unwind_Resume and that exception aborts the interpreter (found with gdb, 2026-10-08).
    return tuple(int(d[i]) for i in range(len(d)))


_TRT2TORCH = None


class TrtImageEncoder(torch.nn.Module):
    """Callable matching NanoOWL's engine wrapper: image (B,3,S,S) CUDA float -> OwlEncodeImageOutput."""

    def __init__(self, engine_path: str | Path, max_batch: int = 1):
        super().__init__()
        import tensorrt as trt
        from nanoowl.owl_predictor import OwlEncodeImageOutput

        self._trt, self._Out, self.max_batch = trt, OwlEncodeImageOutput, max_batch
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise TrtError(f"cannot deserialize {engine_path} (built with a different TensorRT/GPU?)")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        missing = ({"image"} | set(_OUTPUT_NAMES)) - set(names)
        if missing:
            raise TrtError(f"engine lacks tensors {sorted(missing)}")
        dt = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self._dtype = {n: dt[self.engine.get_tensor_dtype(n)] for n in names}

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        if not image.is_cuda:
            raise TrtError("input must be a CUDA tensor")
        outs = []
        for s in range(0, image.shape[0], self.max_batch):
            x = image[s:s + self.max_batch].to(self._dtype["image"]).contiguous()
            self.ctx.set_input_shape("image", tuple(x.shape))
            self.ctx.set_tensor_address("image", x.data_ptr())
            bufs = {}
            for n in _OUTPUT_NAMES:
                bufs[n] = torch.empty(_dims(self.ctx.get_tensor_shape(n)), dtype=self._dtype[n], device=x.device)
                self.ctx.set_tensor_address(n, bufs[n].data_ptr())
            if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise TrtError("execute_async_v3 failed")
            outs.append({n: b.float() for n, b in bufs.items()})  # decode math stays fp32
        return self._Out(**{n: torch.cat([o[n] for o in outs], 0) for n in _OUTPUT_NAMES})
