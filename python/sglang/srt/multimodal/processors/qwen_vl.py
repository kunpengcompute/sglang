import math
import os
import re
import time
from typing import List, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode

from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from sglang.srt.models.qwen2_vl import Qwen2VLForConditionalGeneration
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)
from sglang.srt.utils import is_cpu_920f
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.utils import logger

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
RESIZE_RESAMPLE = getattr(Image, envs.SGLANG_RESIZE_RESAMPLE.get(), None)
if envs.SGLANG_RESIZE_RESAMPLE.is_set() and RESIZE_RESAMPLE is None:
    logger.warning(
        f"Invalid RESIZE_RESAMPLE value: '{envs.SGLANG_RESIZE_RESAMPLE.get()}'. "
        f"Ignoring and using default."
    )
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def _kunpeng_vision_align64_enabled() -> bool:
    """Whether to pre-align images to the Kunpeng ViT flash-attention path.

    Mirrors kunpeng_vit_flash_enabled() in layers/attention/vision.py
    without importing the attention stack into the tokenizer-side process.
    """
    if not is_cpu_920f():
        return False
    return os.environ.get("SGLANG_KUNPENG_VIT_FLASH_ATTN", "1") not in ("", "0")


def kunpeng_vision_align64(img, min_pixels: int, resample=None):
    """Resize a PIL image so its patch grid always takes the flash path.

    The Kunpeng flash kernel needs every image's patch count (side/16)^2 to
    be a multiple of 16 - i.e. both sides multiples of 64 - and within the
    shared flash workspace's row budget (SGLANG_KUNPENG_VISION_SHM_MAX_ROWS
    patches). The stock smart_resize only aligns to patch*merge = 32, which
    drops misaligned or oversized images onto the SDPA math chain where
    TTFT grows quadratically. smart_resize with factor=64 and
    max_pixels = budget * 16 * 16 enforces both gates, and the stock
    factor-32 pass afterwards becomes a no-op because every 64-multiple is
    a 32-multiple.
    """
    if not isinstance(img, Image.Image):
        return img  # tensor / precomputed payloads pass through untouched
    budget = int(os.environ.get("SGLANG_KUNPENG_VISION_SHM_MAX_ROWS", "4096"))
    target_h, target_w = smart_resize(
        img.height,
        img.width,
        factor=64,
        min_pixels=min_pixels,
        max_pixels=budget * 256,
    )
    if (target_h, target_w) != (img.height, img.width):
        logger.info(
            "[align64] %dx%d -> %dx%d (flash rows=%d)",
            img.height,
            img.width,
            target_h,
            target_w,
            budget,
        )
        img = img.resize(
            (target_w, target_h), RESIZE_RESAMPLE or resample or Image.BILINEAR
        )
    return img


def _kunpeng_preproc_profile_enabled() -> bool:
    """One line per image request breaking down the preprocessing stages.

    Diagnostic only - this runs on the frontend PIL/numpy path (kupl-free);
    disable with SGLANG_KUNPENG_PREPROC_PROFILE=0 once the campaign is over.
    """
    if not is_cpu_920f():
        return False
    return os.environ.get("SGLANG_KUNPENG_PREPROC_PROFILE", "1") not in ("", "0")


def _kunpeng_preproc_profile_note(processor, images, load_ms, align_ms, hf_ms):
    """Attribute one request's hf_call budget between pixel math and framework.

    Replays the two numeric stages of the classic HF image processor (PIL
    resize to the factor-32 target; numpy rescale + normalize + transpose)
    on the already-loaded image and prints the split; the remainder of the
    hf_call budget is framework overhead. Read-only: replayed tensors are
    discarded and the request itself is untouched.
    """
    try:
        image_processor = getattr(processor, "image_processor", None)
        mean = np.asarray(
            getattr(image_processor, "image_mean", None) or [0.5, 0.5, 0.5],
            dtype=np.float32,
        )
        std = np.asarray(
            getattr(image_processor, "image_std", None) or [0.5, 0.5, 0.5],
            dtype=np.float32,
        )
        min_pixels = int(
            getattr(image_processor, "min_pixels", 0) or MIN_PIXELS
        )
        max_pixels = int(
            getattr(image_processor, "max_pixels", 0) or MAX_PIXELS
        )
        img = images[0]
        target_h, target_w = smart_resize(
            img.height,
            img.width,
            factor=32,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        t0 = time.perf_counter()
        r_img = img.resize(
            (target_w, target_h), RESIZE_RESAMPLE or Image.BILINEAR
        )
        t_resize = (time.perf_counter() - t0) * 1e3
        arr = np.asarray(r_img, dtype=np.uint8)
        t0 = time.perf_counter()
        x = arr.astype(np.float32) / 255.0
        x = (x - mean) / std
        x = x.transpose(2, 0, 1)
        t_norm = (time.perf_counter() - t0) * 1e3
        logger.info(
            "[kunpeng-preproc] n_img=%d src=%dx%d->%dx%d load=%.1f align64=%.1f"
            " hf_call=%.1f | replay resize=%.1f normalize=%.1f framework~=%.1fms",
            len(images),
            img.height,
            img.width,
            target_h,
            target_w,
            load_ms,
            align_ms,
            hf_ms,
            t_resize,
            t_norm,
            max(hf_ms - t_resize - t_norm, 0.0),
        )
    except Exception as e:
        logger.info("[kunpeng-preproc] profile note skipped (%s)", e)


# ── Fused PIL/numpy pixel_values fast path (Kunpeng 920F frontend) ──────
# The classic (slow, PIL/numpy) image processor spends its hf_call budget
# on framework overhead plus a multi-pass rescale/normalize chain (the
# stage breakdown probe above attributes ~14ms fixed + ~57ms/Mpx to it at
# steady state, vs ~30ms/Mpx for a plain numpy re-implementation). Replace
# the instance's preprocess with a one-pass fused numpy version:
# (x / 255 - mean) / std == x * A + B (A, B pre-assembled per channel),
# the temporal frame repeat, and the Qwen patch flattening order mirrored
# from transform_patches_to_flatten in the NPU fast-path patch. The very
# first fusable input runs BOTH implementations and compares pixel_values
# element-wise (plus image_grid_thw); any mismatch or computational
# surprise falls back to the untouched original for good, while
# non-image inputs (videos, tensors) simply keep the original per call
# and leave the gate armed. Pure PIL/numpy - no torch compute
# op ever runs here, so the kupl thread rules of this frontend process
# are not engaged. Env switch SGLANG_KUNPENG_PREPROC_FASTPATH (default
# on).
def _kunpeng_preproc_fast_enabled() -> bool:
    if not is_cpu_920f():
        return False
    return os.environ.get("SGLANG_KUNPENG_PREPROC_FASTPATH", "1") not in ("", "0")


def _kunpeng_fused_flatten(
    img,
    patch_size,
    temporal_patch_size,
    merge_size,
    min_pixels,
    max_pixels,
    mean,
    std,
    resample,
):
    """One image -> (flat [gh*gw, tp*C*p*p] float32, (1, gh, gw))."""
    target_h, target_w = smart_resize(
        img.height,
        img.width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    if img.mode != "RGB":
        # match do_convert_rgb of the classic processor (which runs before
        # the resize): grayscale and palette images must not leak
        # palette-index or alpha channels into the pixel array
        img = img.convert("RGB")
    if (target_h, target_w) != (img.height, img.width):
        img = img.resize(
            (target_w, target_h), RESIZE_RESAMPLE or resample or Image.BILINEAR
        )
    arr = np.asarray(img, dtype=np.uint8)
    mean3 = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std3 = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    # (x / 255 - mean) / std fused into a single multiply-add pass
    a = (np.float32(1.0) / (np.float32(255.0) * std3)).astype(np.float32)
    b = (-mean3 / std3).astype(np.float32)
    x = arr.astype(np.float32) * a + b  # [H, W, C]
    x = x.transpose(2, 0, 1)  # [C, H, W]
    tp, p, m = temporal_patch_size, patch_size, merge_size
    c = x.shape[0]
    if c == 3 and tp > 1:
        # single frame repeats to fill the temporal patch dimension
        x = np.concatenate([x] * tp, axis=0)  # [tp*C, H, W]
    gh, gw = target_h // p, target_w // p
    # flattening order mirrored from transform_patches_to_flatten (NPU
    # patch), with the batch/temporal-grid leading dims dropped (B == 1)
    v = x.reshape(tp * c, gh // m, m, p, gw // m, m, p)
    v = v.transpose(0, 1, 4, 2, 5, 3, 6)
    v = v.reshape(tp, c, gh * gw, p, p)
    v = v.transpose(2, 1, 0, 3, 4)
    flat = v.reshape(gh * gw, -1)
    return flat, (1, gh, gw)


def _kunpeng_fused_call(image_processor, images, kwargs):
    """Build the BatchFeature the classic preprocess would return."""
    from transformers.image_processing_utils import BatchFeature

    p = image_processor
    patch_size = int(getattr(p, "patch_size", 0) or 16)
    temporal_patch_size = int(getattr(p, "temporal_patch_size", 0) or 2)
    merge_size = int(getattr(p, "merge_size", 0) or 2)
    mean = getattr(p, "image_mean", None) or [0.5, 0.5, 0.5]
    std = getattr(p, "image_std", None) or [0.5, 0.5, 0.5]
    min_pixels = int(getattr(p, "min_pixels", 0) or MIN_PIXELS)
    max_pixels = int(getattr(p, "max_pixels", 0) or MAX_PIXELS)
    resample = kwargs.get("resample") or getattr(p, "resample", None)
    flats = []
    grids = []
    for img in images:
        if not isinstance(img, Image.Image):
            # tensor / precomputed inputs keep the original path
            raise TypeError("non-PIL image input")
        flat, grid = _kunpeng_fused_flatten(
            img,
            patch_size,
            temporal_patch_size,
            merge_size,
            min_pixels,
            max_pixels,
            mean,
            std,
            resample,
        )
        flats.append(flat)
        grids.append(list(grid))
    pixel_values = torch.from_numpy(np.concatenate(flats, axis=0))
    image_grid_thw = torch.tensor(grids, dtype=torch.int64)
    return BatchFeature(
        data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
        tensor_type=kwargs.get("return_tensors"),
    )


def _kunpeng_preproc_ab_diff(out_feat, ref_feat):
    """Element-wise relative diff of the two preprocessing outputs."""
    a = out_feat.get("pixel_values")
    b = ref_feat.get("pixel_values")
    if a is None or b is None:
        return None, "missing pixel_values"
    an = a.detach().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    bn = b.detach().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    shape_note = f"{tuple(bn.shape)}"
    if an.shape != bn.shape:
        if an.size != bn.size:
            return None, f"shape {tuple(an.shape)} vs {tuple(bn.shape)}"
        an = an.reshape(bn.shape)  # leading singleton dims may differ
    ad = an.astype(np.float64)
    bd = bn.astype(np.float64)
    diff = float(np.abs(ad - bd).max())
    denom = float(np.abs(bd).max()) or 1.0
    ga = out_feat.get("image_grid_thw")
    gb = ref_feat.get("image_grid_thw")
    if ga is not None and gb is not None:
        if np.asarray(ga).tolist() != np.asarray(gb).tolist():
            return None, f"grid {np.asarray(ga).tolist()} vs {np.asarray(gb).tolist()}"
    return diff / denom, shape_note


def _kunpeng_patch_image_processor(processor):
    """Instance-level fused fast path for the classic image processor."""
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if type(image_processor).__name__.endswith("Fast"):
        return  # the torchvision backend is not our target (and not used on 920F)
    if getattr(image_processor, "_kunpeng_fused_preprocess", None) is not None:
        return
    orig = image_processor.preprocess
    state = {"checked": False, "use": False, "shape": None}

    def _fused_input(images, kwargs):
        # Only a flat list of PIL images takes the fused path; tensors,
        # nested video frames or a videos kwarg stay with the original.
        if isinstance(images, Image.Image):
            images = [images]
        if kwargs.get("videos") is not None:
            raise TypeError("video input")
        if not isinstance(images, (list, tuple)) or not images:
            raise TypeError("unsupported image input")
        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError("non-PIL image input")
        return images

    def _with_ref_shape(out):
        # The classic path may return a leading batch dim the fused layout
        # does not carry; realign only when the element count agrees.
        shape = state["shape"]
        pv = out["pixel_values"]
        if (
            shape is not None
            and tuple(pv.shape) != shape
            and pv.numel() == int(np.prod(shape))
        ):
            out["pixel_values"] = pv.reshape(shape)
        return out

    def _fused_preprocess(images, *args, **kwargs):
        if state["use"]:
            try:
                imgs = _fused_input(images, kwargs)
                if args:
                    raise TypeError("unexpected positional args")
                return _with_ref_shape(
                    _kunpeng_fused_call(image_processor, imgs, kwargs)
                )
            except Exception as e:
                # steady-state surprise (non-PIL input, video frames, an
                # unsupported kwarg, ...): permanently fall back to the
                # untouched original
                state["use"] = False
                logger.info(
                    "[kunpeng-preproc-fast] 稳态输入不支持（%s） -> 永久回退", e
                )
                return orig(images, *args, **kwargs)
        if not state["checked"]:
            # A non-fusable input (e.g. a video) must not burn the one-shot
            # A/B check - quietly stay on the original for this call.
            try:
                imgs = _fused_input(images, kwargs)
                if args:
                    raise TypeError("unexpected positional args")
            except TypeError:
                return orig(images, *args, **kwargs)
            # First real fusable input: run BOTH implementations and gate
            # the fast path on an element-wise comparison (plus the grid).
            state["checked"] = True
            ref = orig(images, *args, **kwargs)
            try:
                out = _kunpeng_fused_call(image_processor, imgs, kwargs)
            except Exception as e:
                logger.info(
                    "[kunpeng-preproc-fast] fused 实现异常（%s） -> 回退", e
                )
                return ref
            rel, note = _kunpeng_preproc_ab_diff(out, ref)
            n_img = len(imgs)
            if rel is not None and rel <= 1e-3:
                state["use"] = True
                b = ref.get("pixel_values")
                if (
                    isinstance(b, torch.Tensor)
                    and tuple(b.shape) != tuple(out["pixel_values"].shape)
                ):
                    state["shape"] = tuple(b.shape)
                out = _with_ref_shape(out)
                logger.info(
                    "[kunpeng-preproc-fast] n_img=%d A/B err=%.1e %s -> 启用",
                    n_img,
                    rel,
                    note,
                )
                return out
            logger.info(
                "[kunpeng-preproc-fast] n_img=%d A/B 失败（%s） -> 回退",
                n_img,
                note if rel is None else f"err={rel:.1e} {note}",
            )
            return ref
        return orig(images, *args, **kwargs)

    image_processor._kunpeng_fused_preprocess = _fused_preprocess
    image_processor.preprocess = _fused_preprocess


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


# process video, qwen-specific
async def preprocess_video(
    vr,
    image_factor: int = IMAGE_FACTOR,
    video_config: dict = {},
) -> torch.Tensor:
    # preprocessed video
    is_video_obj = isinstance(vr, VideoDecoderWrapper)
    if not is_video_obj:
        return vr, None
    entry_time = time.perf_counter()

    total_frames, video_fps = len(vr), vr.avg_fps

    nframes = smart_nframes(
        video_config, total_frames=total_frames, video_fps=video_fps
    )
    idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
    idx = np.unique(idx)

    video = vr.get_frames_as_tensor(idx.tolist())

    video = video.permute(0, 3, 1, 2)  # NHWC -> TCHW

    nframes, _, height, width = video.shape
    min_pixels = video_config.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = video_config.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(
            video_config.get("max_pixels", VIDEO_MAX_PIXELS),
            total_pixels / nframes * FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )

    get_batch_time = time.perf_counter()

    max_pixels_supposed = video_config.get("max_pixels", max_pixels)

    if max_pixels_supposed > max_pixels:
        logger.warning(
            f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}]."
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in video_config and "resized_width" in video_config:
        resized_height, resized_width = smart_resize(
            video_config["resized_height"],
            video_config["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    smart_resize_time = time.perf_counter()
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
    )
    video = video.pin_memory()
    video_metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": idx,
        "video_backend": "torchvision",
    }
    torchvision_resize_time = time.perf_counter()
    logger.debug(
        f"[preprocess_video Perf], "
        f"get_batch_time: {(get_batch_time - entry_time) * 1000:.2f} ms, "
        f"smart_resize_time: {(smart_resize_time - get_batch_time) * 1000:.2f} ms, "
        f"torchvision_resize_time: {(torchvision_resize_time - smart_resize_time) * 1000:.2f} ms, "
        f"total_time: {(torchvision_resize_time - entry_time) * 1000:.2f} ms"
    )
    return video, video_metadata


# Compatible with Qwen-VL & Qwen-Omni Series
class QwenVLImageProcessor(SGLangBaseProcessor):
    supports_transformers_backend = True
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type
        if hf_config.model_type == "qwen3_omni_moe":
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id

        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", None)

        self.audio_start_token_id = getattr(hf_config, "audio_start_token_id", None)
        self.audio_token_id = getattr(hf_config, "audio_token_id", None)

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            # The regex that matches expanded image tokens.
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=self.VIDEO_TOKEN_ID,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        # Kunpeng 920F: swap the classic (PIL/numpy) image processor's
        # numeric stage for the fused one-pass version. The swap is
        # instance-level and gated by a first-input A/B - see
        # _kunpeng_patch_image_processor for the gate and the fallback
        # rules.
        if _kunpeng_preproc_fast_enabled():
            _kunpeng_patch_image_processor(self._processor)

    def build_input_ids_with_timestamps(
        self, prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
    ):
        """
        Build input_ids with timestamps for qwen3_vl models.
        """
        if not isinstance(prompt, list):
            prompt = self._processor.tokenizer.encode(prompt)

        img_token_id = getattr(self, "IM_TOKEN_ID", None)
        video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
        audio_token_id = getattr(self, "audio_token_id", None)
        spatial_merge_size = getattr(self, "spatial_merge_size", 1)
        vision_start_token_id = getattr(self, "vision_start_token_id", None)
        vision_end_token_id = getattr(self, "vision_end_token_id", None)

        input_ids = []
        offsets = []
        modality_list = []
        cur_idx = 0

        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if img_token_id is not None and prompt[i + 1] == img_token_id:
                vision_start_indices.append((i, Modality.IMAGE))
            elif video_token_id is not None and prompt[i + 1] == video_token_id:
                vision_start_indices.append((i, Modality.VIDEO))

        img_idx = 0
        video_idx = 0
        model_type = getattr(self, "model_type", None)
        for mm_start_idx, modality in vision_start_indices:
            modality_list.append(modality)
            video_tokens = None
            if modality == Modality.IMAGE:
                mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)
                mm_token_id = img_token_id
                img_idx += 1
            elif modality == Modality.VIDEO:
                curr_timestamps = video_timestamps[video_idx]
                num_frames = video_grid_thw[video_idx][0]
                frame_seqlen = video_grid_thw[video_idx][1:].prod().item() // (
                    spatial_merge_size**2
                )
                video_tokens = []
                _current_offset = len(input_ids) + mm_start_idx + 1 - cur_idx
                # take single frame as one mm_item
                for frame_idx in range(num_frames):
                    if frame_idx > 0:
                        modality_list.append(Modality.VIDEO)
                    curr_time = curr_timestamps[frame_idx]
                    timestamp_text = f"<{curr_time:.1f} seconds>"
                    timestamp_tokens = self._processor.tokenizer.encode(
                        timestamp_text, add_special_tokens=False
                    )
                    video_tokens.extend(timestamp_tokens)
                    _current_offset += len(timestamp_tokens)
                    if vision_start_token_id is not None:
                        video_tokens.append(vision_start_token_id)
                        _current_offset += 1
                    video_tokens.extend([video_token_id] * frame_seqlen)
                    if vision_end_token_id is not None:
                        video_tokens.append(vision_end_token_id)
                    offsets.append(
                        (_current_offset, _current_offset + frame_seqlen - 1)
                    )
                    _current_offset += (
                        frame_seqlen + 1
                        if vision_end_token_id is not None
                        else frame_seqlen
                    )  # for vision_end_token_id
                mm_token_num = len(video_tokens)
                mm_token_id = None
                video_idx += 1
            else:
                logger.warning(
                    f"{modality} modality is not supported for qwen3_vl models with timestamps."
                )
                continue
            assert cur_idx <= mm_start_idx
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            if modality == Modality.VIDEO:
                input_ids.extend(video_tokens)
            else:
                mm_offset_start = len(input_ids)
                input_ids.extend([mm_token_id] * mm_token_num)
                offsets.append((mm_offset_start, len(input_ids) - 1))
            cur_idx = mm_start_idx + 2  # jump to vision_end_id
        else:
            input_ids.extend(prompt[cur_idx:])

        return input_ids, offsets, modality_list

    def compute_mrope_positions(self, input_ids, mm_items):
        image_grid_thw = None
        video_grid_thw = None
        for item in mm_items:
            if "image_grid_thw" in item.model_specific_data:
                image_grid_thw = item.model_specific_data["image_grid_thw"]
            if "video_grid_thw" in item.model_specific_data:
                video_grid_thw = item.model_specific_data["video_grid_thw"]

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        audio_feature_lens = kwargs.get("audio_feature_lens", None)
        video_timestamps = kwargs.get("video_timestamps", None)
        second_per_grid_ts = kwargs.get("second_per_grid_ts", None)

        audio_seq_lens = None
        if audio_feature_lens is not None:
            if self.model_type == "qwen3_omni_moe":
                # apply _get_feat_extract_lengths to get seq_lens
                input_lengths_leave = audio_feature_lens % 100
                feat_lengths = (input_lengths_leave - 1) // 2 + 1
                audio_seq_lens = (
                    ((feat_lengths - 1) // 2 + 1 - 1) // 2
                    + 1
                    + (audio_feature_lens // 100) * 13
                )
            elif self.model_type == "qwen2_5_omni":
                audio_seq_lens = (audio_feature_lens - 1) // 2 + 1
                audio_seq_lens = (audio_seq_lens - 2) // 2 + 1

        if (
            self.model_type in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
            and video_timestamps is not None
        ):
            input_ids, offsets, modality_list = self.build_input_ids_with_timestamps(
                prompt, embeddings, img_grid_thw, video_grid_thw, video_timestamps
            )
        else:
            input_ids, offsets, modality_list = self.build_input_ids(
                prompt, img_grid_thw, video_grid_thw, audio_seq_lens=audio_seq_lens
            )
        assert all(isinstance(modality, Modality) for modality in modality_list)

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            input_ids=torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
            image_grid_thw=img_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_audio_in_video=False,
            audio_seqlens=(
                audio_feature_lens if self.model_type == "qwen3_omni_moe" else None
            ),
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_token_id=self.audio_start_token_id,
            position_id_per_seconds=getattr(
                self.hf_config, "position_id_per_seconds", None
            ),
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
        )
        mrope_positions = mrope_positions.squeeze(1)

        mm_items = []
        consumed_per_modality = {}

        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            embedding_start = consumed_per_modality.get(modality, 0)
            embedding_slice = embeddings[modality][
                embedding_start : embedding_start + num_tokens
            ]
            consumed_per_modality[modality] = embedding_start + num_tokens
            logger.info(f"Get embedding slice for {modality}, num_tokens={num_tokens}")
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embedding_slice,
                )
            )

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        entry_time = time.perf_counter()
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        load_time = time.perf_counter()

        # Kunpeng 920F: pre-resize images to 64-aligned dims within the
        # flash row budget, before the HF processor's factor-32 smart_resize
        # re-rounds them onto the SDPA path (see kunpeng_vision_align64).
        # This is the single choke point above both load paths (fast and
        # legacy) that every image request through this processor passes.
        if base_output.images and _kunpeng_vision_align64_enabled():
            image_processor = getattr(self._processor, "image_processor", None)
            min_pixels = int(getattr(image_processor, "min_pixels", 0) or 65_536)
            resample = getattr(image_processor, "resample", None)
            _t_align = time.perf_counter()
            base_output.images = [
                kunpeng_vision_align64(img, min_pixels, resample)
                for img in base_output.images
            ]
            align_ms = (time.perf_counter() - _t_align) * 1e3
        else:
            align_ms = 0.0

        rid = getattr(request_obj, "rid", "anonymous_rid")

        video_metadata = None
        if base_output.videos:
            videos_processed = [
                await preprocess_video(video, video_config=self.video_config)
                for video in base_output.videos
            ]
            base_output.videos, video_metadata = map(list, zip(*videos_processed))

        preprocess_time = time.perf_counter()

        # NOTE: for qwen3-vl, video_meta need to be passed in, since do_sample_frames is already done in preprocess_video
        _t_hf = time.perf_counter()
        if self.hf_config.model_type in (
            "qwen3_vl",
            "qwen3_vl_moe",
            "qwen3_5",
            "qwen3_5_moe",
        ):
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output,
                self.mm_tokens,
                video_metadata=video_metadata,
                do_sample_frames=False,
            )
        else:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )
        _hf_ms = (time.perf_counter() - _t_hf) * 1e3
        if base_output.images and _kunpeng_preproc_profile_enabled():
            _kunpeng_preproc_profile_note(
                self._processor,
                base_output.images,
                (load_time - entry_time) * 1e3,
                align_ms,
                _hf_ms,
            )

        audio_feature_lengths = None

        if self.model_type == "qwen3_omni_moe":
            audio_item = next((mm for mm in mm_items if mm.is_audio()), None)
            if audio_item:
                audio_feature_lengths = torch.sum(
                    audio_item.feature_attention_mask, dim=1
                )

        second_per_grid_ts = getattr(ret, "second_per_grid_ts", None)
        if second_per_grid_ts is None:
            second_per_grid_ts = getattr(ret, "video_second_per_grid", None)

        process_time = time.perf_counter()

        input_ids = input_ids.flatten()

        image_grid_thw = None
        if hasattr(ret, "image_grid_thw"):
            image_grid_thw = ret.image_grid_thw

        if image_grid_thw is None and image_data and isinstance(image_data[0], dict):
            image_grid_thw = image_data[0].get("image_grid_thw")

        video_grid_thw = None
        if hasattr(ret, "video_grid_thw"):
            video_grid_thw = ret.video_grid_thw

        if video_grid_thw is None and request_obj.video_data:
            first_video = request_obj.video_data[0]
            if isinstance(first_video, dict):
                video_grid_thw = first_video.get("video_grid_thw")

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            # use the expanded token ids
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
            video_grid_thw=getattr(ret, "video_grid_thw", None),
            second_per_grid_ts=second_per_grid_ts,
            use_audio_in_video=False,
            audio_seqlens=audio_feature_lengths,
            audio_token_id=getattr(self.hf_config, "audio_token_id", None),
            audio_start_token_id=self.audio_start_token_id,
            position_id_per_seconds=getattr(
                self.hf_config, "position_id_per_seconds", None
            ),
        )
        mrope_positions = mrope_positions.squeeze(1)
        get_rope_index_time = time.perf_counter()
        logger.debug(
            f"[QwenVLProcessor Perf] {rid=}, "
            f"load_time: {(load_time - entry_time) * 1000:.2f} ms, "
            f"preprocess_time: {(preprocess_time - load_time) * 1000:.2f} ms, "
            f"process_time: {(process_time - preprocess_time) * 1000:.2f} ms, "
            f"get_rope_index_time: {(get_rope_index_time - process_time) * 1000:.2f} ms, "
            f"total_time: {(get_rope_index_time - entry_time) * 1000:.2f} ms"
        )

        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
