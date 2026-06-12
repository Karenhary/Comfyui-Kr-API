import base64
import binascii
import contextvars
import io
import json
import math
import os
import random
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error as url_error
from urllib import request as url_request
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image

_BYPASS_PROXY_CTX: contextvars.ContextVar[bool] = contextvars.ContextVar("kr_bypass_proxy", default=True)


def _set_bypass_proxy(enabled: bool) -> None:
    _BYPASS_PROXY_CTX.set(bool(enabled))


def _is_bypass_proxy() -> bool:
    try:
        return bool(_BYPASS_PROXY_CTX.get())
    except Exception:
        return True


def _requests_proxy_kwargs() -> Dict[str, Any]:
    if _is_bypass_proxy():
        return {"proxies": {"http": None, "https": None}}
    return {}


def _http_get(url: str, **kwargs) -> requests.Response:
    return requests.get(url, **_requests_proxy_kwargs(), **kwargs)


def _http_post(url: str, **kwargs) -> requests.Response:
    return requests.post(url, **_requests_proxy_kwargs(), **kwargs)

try:
    from comfy_api.latest import VideoFromFile
except Exception:
    VideoFromFile = None

try:
    from comfy.comfy_types import IO
except Exception:
    class _IOFallback:
        VIDEO = "VIDEO"

    IO = _IOFallback()

BASE_URL = "https://ai.krapi.cn/v1"
CHAT_COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
IMAGE_GENERATIONS_URL = f"{BASE_URL}/images/generations"
OPENAI_API_ROOT = "https://ai.krapi.cn/"
OPENAI_API_V1 = OPENAI_API_ROOT.rstrip("/") + "/v1"
OPENAI_IMAGE_GENERATIONS_URL = f"{OPENAI_API_V1}/images/generations"
OPENAI_IMAGE_EDITS_URL = f"{OPENAI_API_V1}/images/edits"
VEO_VIDEO_CREATE_URL = OPENAI_API_ROOT.rstrip("/") + "/v1/videos"
VEO_VIDEO_QUERY_URL = OPENAI_API_ROOT.rstrip("/") + "/v1/videos"
GROK_API_ROOT = "https://ai.krapi.cn/"
GROK_API_V1 = GROK_API_ROOT.rstrip("/") + "/v1"
GROK_VIDEO_CREATE_URL = GROK_API_V1 + "/videos"
GROK_VIDEO_QUERY_URL = GROK_API_V1 + "/videos"
GROK_VIDEO_LEGACY_CREATE_URL = GROK_API_V1 + "/video/create"
GROK_VIDEO_LEGACY_QUERY_URL = GROK_API_V1 + "/video/query"
KLING_IMAGE2VIDEO_CREATE_URL = OPENAI_API_ROOT.rstrip("/") + "/kling/v1/videos/image2video"
KLING_IMAGE2VIDEO_QUERY_URL = OPENAI_API_ROOT.rstrip("/") + "/kling/v1/videos/image2video"

CATEGORY_NAME = "\u004b\u0052 API\u4e2d\u8f6c"
AUTO_LABEL = "Auto"

LLM_MODEL_PRESETS = [
    "【R】gemini-3-pro-preview",
    "gemini-3-pro-preview",
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-3-opus-20240229",
    "gemini-1.5-pro-latest",
]

GEMINI_IMAGE_MODEL_PRESETS = [
    "【X】gemini-3.1-flash-image-preview",
]

GEMINI_ASPECT_RATIO_OPTIONS = [
    AUTO_LABEL,
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "5:4",
    "4:5",
    "21:9",
    "1:4",
    "4:1",
    "1:8",
    "8:1",
]

GEMINI_STREAM_ASPECT_RATIO_OPTIONS = [
    "自动",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "1:8",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
]

GEMINI_IMAGE_SIZE_PRESETS = [
    "1K",
    "2K",
    "4K",
]

OPENAI_IMAGE_MODEL_PRESETS = [
    "GPT-Image2-1k",
    "GPT-Image2-2k",
    "GPT-Image2-4k",
]

OPENAI_IMAGE_RESOLUTION_PRESETS = [
    "1k",
    "2k",
    "4k",
]

OPENAI_IMAGE_ASPECT_RATIO_OPTIONS = [
    "自动",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]

# ----------------------------------------------------------------------------
# 模型 → 允许比例 白名单（必须和 server.js 中 normalizeOnlyNineNewApiImageModels 保持一致）
# server.js 会按这张表把不合法的比例就近回退，客户端这里也做一份，
# 让用户在 ComfyUI 里选了不支持的比例时能立刻看到日志，并自动落到最接近的合法比例。
# ----------------------------------------------------------------------------
KR_MODEL_RATIO_WHITELIST: Dict[str, List[str]] = {
    # GPT-Image2 系列
    "gpt-image2": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
    # Nano-Banana2 系列（多了 1:8 / 8:1）
    "nano-banana2": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "1:8", "8:1", "9:16", "16:9", "21:9"],
    # Nano-Banana-Pro（不支持 2:3 / 3:2 / 1:8 / 8:1）
    "nano-banana-pro": ["1:1", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
}


def _kr_model_base_for_ratio_check(model_name: str) -> Optional[str]:
    """提取一个模型预设里的 base 部分，例如 Nano-Banana-Pro-2k → nano-banana-pro。
    无法识别的模型返回 None，调用方就直接放行（不会限制）。"""
    text = (model_name or "").strip().lower()
    if not text:
        return None
    text = text.replace("_", "-").replace("–", "-").replace("—", "-").replace("－", "-")
    text = re.sub(r"\s+", "", text)
    m = re.match(r"^(gpt-image2|nano-banana-pro|nano-banana2)(?:-(?:1k|2k|4k))?$", text)
    if m:
        return m.group(1)
    return None


def _ratio_to_float_kr(ratio: str) -> Optional[float]:
    if not ratio:
        return None
    parts = re.split(r"[:xX]", ratio.strip())
    if len(parts) != 2:
        return None
    try:
        a = float(parts[0])
        b = float(parts[1])
    except ValueError:
        return None
    if a <= 0 or b <= 0:
        return None
    return a / b


def _kr_pick_closest_allowed_ratio(requested: str, allowed: List[str]) -> str:
    """在 allowed 列表里找跟 requested 最接近的比例（按 log 距离比较）。"""
    if not allowed:
        return "1:1"
    req_num = _ratio_to_float_kr(requested)
    if req_num is None:
        return "1:1" if "1:1" in allowed else allowed[0]
    best = allowed[0]
    best_score = float("inf")
    for r in allowed:
        n = _ratio_to_float_kr(r)
        if n is None:
            continue
        score = abs(math.log(req_num / n))
        if score < best_score:
            best_score = score
            best = r
    return best


def _kr_constrain_ratio_for_model(model_name: str, ratio: str) -> Tuple[str, Optional[str]]:
    """根据模型预设把比例限制到允许范围内。
    返回 (final_ratio, warning_message_or_None)。
    无法识别的模型直接放行。"""
    base = _kr_model_base_for_ratio_check(model_name)
    if not base:
        return ratio, None
    allowed = KR_MODEL_RATIO_WHITELIST.get(base)
    if not allowed:
        return ratio, None
    if ratio in allowed:
        return ratio, None
    fallback = _kr_pick_closest_allowed_ratio(ratio, allowed)
    msg = f"model {model_name} does not support ratio {ratio}, fallback to {fallback}"
    _log(f"[比例校验] {msg}")
    return fallback, msg

VEO_MODEL_PRESETS = [
    "veo_3_1_i2v_fast_landscape",
    "veo_3_1_i2v_fast_portrait",
    "veo_3_1_i2v_fast_landscape_1080p",
    "veo_3_1_i2v_fast_portrait_4k",
    "veo_3_1_i2v_s_landscape",
    "veo_3_1_i2v_s_landscape_1080p",
    "veo_3_1_i2v_s_portrait_4k",
    "veo_3_1_i2v_lite_portrait",
]

GROK_VIDEO_MODEL_PRESETS = [
    "grok-videos",
]

KLING_VIDEO_MODEL_PRESETS = [
    "kling-v2-1",
    "kling-v2-1-master",
    "kling-v2-master",
    "kling-v1-6",
    "kling-v1-5",
    "kling-v1",
    "kling-v2-5-turbo",
    "kling-v2-6",
    "kling-v3",
]

GROK_ASPECT_RATIO_OPTIONS = [
    "9:16",
    "16:9",
]

KLING_DURATION_OPTIONS = [
    "5秒",
    "10秒",
]

VEO_ASPECT_RATIO_OPTIONS = [
    "16:9",
    "9:16",
]

VEO_RESOLUTION_OPTIONS = [
    "720p",
    "1080p",
    "4k",
]

VEO_DURATION_OPTIONS = [
    "4秒",
    "6秒",
    "8秒",
]

GEMINI_ASYNC_SUBMISSION_QUEUE: List[Dict[str, Any]] = []
GEMINI_ASYNC_TASK_QUEUE: List[Dict[str, Any]] = []
GEMINI_ASYNC_QUEUE_LOCK = threading.Lock()
GEMINI_ASYNC_WORKERS_STARTED = False
GEMINI_ASYNC_WORKER_COUNT = 3

OPENAI_IMAGE_ASYNC_SUBMISSION_QUEUE: List[Dict[str, Any]] = []
OPENAI_IMAGE_ASYNC_TASK_QUEUE: List[Dict[str, Any]] = []
OPENAI_IMAGE_ASYNC_QUEUE_LOCK = threading.Lock()
OPENAI_IMAGE_ASYNC_WORKERS_STARTED = False
OPENAI_IMAGE_ASYNC_WORKER_COUNT = 3


def _log(message: str) -> None:
    print(f"[Comfyui-Kr-API] {message}")


def _blank_image(size: int = 512) -> torch.Tensor:
    return torch.zeros((1, size, size, 3), dtype=torch.float32)


def _resolve_veo_size(aspect_ratio: str, resolution: str) -> str:
    ratio = (aspect_ratio or "16:9").strip()
    res = (resolution or "1080p").strip().lower()
    mapping = {
        ("16:9", "720p"): "1280x720",
        ("16:9", "1080p"): "1920x1080",
        ("16:9", "4k"): "3840x2160",
        ("9:16", "720p"): "720x1280",
        ("9:16", "1080p"): "1080x1920",
        ("9:16", "4k"): "2160x3840",
    }
    return mapping.get((ratio, res), "1920x1080")


def _parse_veo_duration_seconds(duration_label: str) -> int:
    label = (duration_label or "").strip()
    m = re.search(r"\d+", label)
    if m:
        value = int(m.group(0))
        if value in (4, 6, 8):
            return value
    return 6


def _resolve_grok_video_size(aspect_ratio: str) -> str:
    ratio = (aspect_ratio or "9:16").strip()
    if ratio in {"9:16", "16:9"}:
        return ratio
    return "9:16"


def _parse_grok_duration_seconds(duration_label: str) -> int:
    label = (duration_label or "").strip()
    m = re.search(r"\d+", label)
    if m:
        value = int(m.group(0))
        if value in (6, 10):
            return value
    return 10


def _parse_kling_duration_seconds(duration_label: str) -> int:
    label = (duration_label or "").strip()
    m = re.search(r"\d+", label)
    if m:
        value = int(m.group(0))
        if value in (5, 10):
            return value
    return 5


class KRVideoAdapter:
    def __init__(self, video_path_or_url: str = ""):
        source = (video_path_or_url or "").strip()
        self.source = source
        self.is_url = source.startswith(("http://", "https://"))
        self.video_url = source if self.is_url else ""
        self.video_path = source if source and not self.is_url else ""

    def get_dimensions(self):
        return 1280, 720

    def save_to(self, output_path, format="auto", codec="auto", metadata=None):
        if not self.source:
            return False
        try:
            if self.is_url:
                response = _http_get(self.video_url, stream=True, timeout=(30, 300))
                response.raise_for_status()
                with open(output_path, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                return True

            shutil.copyfile(self.video_path, output_path)
            return True
        except Exception as exc:
            _log(f"KRVideoAdapter.save_to failed: {exc}")
            return False


def _resolve_model_name(model_preset: str, custom_model: str) -> str:
    custom = (custom_model or "").strip()
    return custom if custom else (model_preset or "").strip()


def _sanitize_model_name_for_provider(model_name: str) -> str:
    text = (model_name or "").strip()
    if not text:
        return ""
    # Strip UI markers like "【X】", "【R】", "[X]" that should not be sent upstream.
    text = re.sub(r"^\s*[【\[]\s*[A-Za-z]\s*[】\]]\s*", "", text)
    return text.strip()


def _make_headers(api_key: str) -> Dict[str, str]:
    key = (api_key or "").strip()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "X-API-Key": key,
        "X-Banana-Client": "comfyui-kr-api",
    }


def _make_multipart_headers(api_key: str) -> Dict[str, str]:
    headers = _make_headers(api_key)
    headers.pop("Content-Type", None)
    return headers


def _make_openai_compat_headers(api_key: str) -> Dict[str, str]:
    key = (api_key or "").strip()
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }


def _kr_candidate_urls(url: str) -> List[str]:
    urls = [str(url or "").strip()]
    if urls[0].startswith("https://"):
        urls.append("http://" + urls[0][len("https://"):])
    return [u for u in urls if u]


def _kr_post_json_with_fallback(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: Any):
    last_err = None
    for target_url in _kr_candidate_urls(url):
        try:
            with requests.Session() as session:
                return session.post(target_url, json=payload, headers=headers, timeout=timeout, verify=False)
        except requests.RequestException as exc:
            last_err = exc
    if last_err:
        raise last_err
    raise RuntimeError("POST failed without response")


def _kr_post_form_with_fallback(url: str, headers: Dict[str, str], data: Dict[str, Any], files: Optional[List[Tuple[str, Any]]], timeout: Any):
    last_err = None
    for target_url in _kr_candidate_urls(url):
        try:
            with requests.Session() as session:
                return session.post(
                    target_url,
                    headers=headers,
                    data=data,
                    files=files if files else None,
                    timeout=timeout,
                    verify=False,
                    allow_redirects=False,
                )
        except requests.RequestException as exc:
            last_err = exc
    if last_err:
        raise last_err
    raise RuntimeError("POST form failed without response")


def _kr_get_with_fallback(url: str, headers: Dict[str, str], timeout: Any):
    last_err = None
    for target_url in _kr_candidate_urls(url):
        try:
            with requests.Session() as session:
                return session.get(target_url, headers=headers, timeout=timeout, verify=False)
        except requests.RequestException as exc:
            last_err = exc
    if last_err:
        raise last_err
    raise RuntimeError("GET failed without response")


def _kr_post_json_direct(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: Any):
    with requests.Session() as session:
        return session.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
            verify=False,
            allow_redirects=False,
        )


def _kr_post_form_direct(url: str, headers: Dict[str, str], data: Dict[str, Any], files: Optional[List[Tuple[str, Any]]], timeout: Any):
    with requests.Session() as session:
        return session.post(
            url,
            headers=headers,
            data=data,
            files=files if files else None,
            timeout=timeout,
            verify=False,
            allow_redirects=False,
        )


def _kr_get_direct(url: str, headers: Dict[str, str], timeout: Any, params: Optional[Dict[str, Any]] = None):
    with requests.Session() as session:
        return session.get(
            url,
            headers=headers,
            params=params,
            timeout=timeout,
            verify=False,
            allow_redirects=False,
        )


def _is_write_timeout_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return ("write operation timed out" in text) or ("connection aborted" in text and "timed out" in text)


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    tensor = image_tensor[0] if image_tensor.dim() == 4 else image_tensor
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)

    if tensor.dim() != 3:
        raise ValueError(f"Unsupported tensor shape: {tuple(tensor.shape)}")

    if tensor.shape[-1] in (3, 4):
        hwc = tensor[..., :3]
    elif tensor.shape[0] in (3, 4):
        hwc = tensor[:3].permute(1, 2, 0)
    else:
        raise ValueError(f"Unsupported image channel shape: {tuple(tensor.shape)}")

    arr = (hwc.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _tensor_to_data_url(image_tensor: torch.Tensor) -> str:
    image = _tensor_to_pil(image_tensor).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=100)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _tensor_to_inline_data(image_tensor: torch.Tensor) -> Dict[str, str]:
    image = _tensor_to_pil(image_tensor).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=100)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "mimeType": "image/jpeg",
        "data": encoded,
    }


def _tensor_to_base64(image_tensor: torch.Tensor) -> str:
    image = _tensor_to_pil(image_tensor).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=100)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _tensor_to_base64_jpeg(image_tensor: torch.Tensor, quality: int = 100) -> str:
    image = _tensor_to_pil(image_tensor).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _download_image_to_tensor(url: str, timeout_sec: int = 90) -> torch.Tensor:
    response = _http_get(url, timeout=timeout_sec)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return _pil_to_tensor(image)


def _decode_base64_image_to_tensor(data: str) -> Optional[torch.Tensor]:
    try:
        raw = base64.b64decode(data)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return _pil_to_tensor(image)
    except Exception:
        return None


def _download_image_to_base64(url: str, timeout_sec: int = 30) -> Optional[str]:
    try:
        response = _http_get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/*,*/*;q=0.8",
            },
            timeout=timeout_sec,
        )
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")
    except Exception:
        return None


def _download_video_to_temp(video_url: str, timeout_sec: int = 600, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
    url = (video_url or "").strip()
    if not url:
        return None
    try:
        response = _http_get(url, stream=True, headers=headers or {}, timeout=(30, timeout_sec))
        response.raise_for_status()
        tmp_dir = os.path.join(tempfile.gettempdir(), "comfyui_kr_api_videos")
        os.makedirs(tmp_dir, exist_ok=True)
        ext = ".mp4"
        lower = url.lower()
        if ".webm" in lower:
            ext = ".webm"
        elif ".mov" in lower:
            ext = ".mov"
        file_path = os.path.join(tmp_dir, f"veo_{int(time.time() * 1000)}{ext}")
        with open(file_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        return file_path
    except Exception as exc:
        _log(f"video download failed, fallback to remote url: {exc}")
        return None


def _download_video_content_by_id(api_key: str, task_id: str, timeout_sec: int = 600) -> Optional[str]:
    task = (task_id or "").strip()
    if not task:
        return None
    content_url = f"{OPENAI_API_V1}/videos/{task}/content"
    try:
        response = _http_get(content_url, headers={"Authorization": f"Bearer {(api_key or '').strip()}"}, stream=True, timeout=(30, timeout_sec))
        if response.status_code != 200:
            _log(f"video content endpoint failed: http={response.status_code}, body={(response.text or '')[:220]}")
            return None
        tmp_dir = os.path.join(tempfile.gettempdir(), "comfyui_kr_api_videos")
        os.makedirs(tmp_dir, exist_ok=True)
        file_path = os.path.join(tmp_dir, f"veo_content_{int(time.time() * 1000)}.mp4")
        with open(file_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        return file_path
    except Exception as exc:
        _log(f"video content download failed: {exc}")
        return None


def _download_grok_video_content_by_id(api_key: str, task_id: str, timeout_sec: int = 600) -> Optional[str]:
    task = (task_id or "").strip()
    if not task:
        return None
    content_url = f"{GROK_VIDEO_QUERY_URL}/{task}/content"
    try:
        response = _http_get(
            content_url,
            headers={"Authorization": f"Bearer {(api_key or '').strip()}"},
            stream=True,
            timeout=(30, timeout_sec),
        )
        if response.status_code != 200:
            _log(f"Grok video content endpoint failed: http={response.status_code}, body={(response.text or '')[:220]}")
            return None
        tmp_dir = os.path.join(tempfile.gettempdir(), "comfyui_kr_api_videos")
        os.makedirs(tmp_dir, exist_ok=True)
        file_path = os.path.join(tmp_dir, f"grok_content_{int(time.time() * 1000)}.mp4")
        with open(file_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        return file_path
    except Exception as exc:
        _log(f"Grok video content download failed: {exc}")
        return None


def _extract_file_id_from_payload(payload: Any) -> Optional[str]:
    file_ids: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).lower()
                if isinstance(value, str):
                    text = value.strip()
                    if text.startswith("file-"):
                        file_ids.append(text)
                    else:
                        text_norm = text.replace("\\/", "/")
                        for hit in re.findall(r"file-[A-Za-z0-9_-]+", text_norm):
                            file_ids.append(hit)
                elif isinstance(value, list) and key_l in {"output", "outputs", "files", "file_ids", "data"}:
                    for item in value:
                        if isinstance(item, str) and item.strip().startswith("file-"):
                            file_ids.append(item.strip())
                        elif isinstance(item, dict):
                            fid = item.get("id") or item.get("file_id")
                            if isinstance(fid, str) and fid.strip().startswith("file-"):
                                file_ids.append(fid.strip())
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return file_ids[0] if file_ids else None


def _download_video_content_by_file_id(api_key: str, file_id: str, timeout_sec: int = 600) -> Optional[str]:
    fid = (file_id or "").strip()
    if not fid:
        return None
    content_url = f"{GROK_API_V1}/files/{fid}/content"
    try:
        response = _http_get(
            content_url,
            headers={"Authorization": f"Bearer {(api_key or '').strip()}"},
            stream=True,
            timeout=(30, timeout_sec),
        )
        if response.status_code != 200:
            _log(f"video file content endpoint failed: http={response.status_code}, body={(response.text or '')[:220]}")
            return None
        tmp_dir = os.path.join(tempfile.gettempdir(), "comfyui_kr_api_videos")
        os.makedirs(tmp_dir, exist_ok=True)
        file_path = os.path.join(tmp_dir, f"grok_file_{int(time.time() * 1000)}.mp4")
        with open(file_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        return file_path
    except Exception as exc:
        _log(f"video file content download failed: {exc}")
        return None


def _tensor_to_compressed_jpeg_data_url(image_tensor: torch.Tensor, max_long_side: int = 1024, quality: int = 100) -> str:
    frame = image_tensor[0:1] if image_tensor.dim() == 4 else image_tensor.unsqueeze(0)
    image = _tensor_to_pil(frame).convert("RGB")
    w, h = image.size
    long_side = max(w, h)
    if long_side > max_long_side and long_side > 0:
        scale = float(max_long_side) / float(long_side)
        new_w = max(16, int(round(w * scale)))
        new_h = max(16, int(round(h * scale)))
        resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image = image.resize((new_w, new_h), resampling)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _tensor_to_compressed_jpeg_bytes(image_tensor: torch.Tensor, max_long_side: int = 2048, quality: int = 100) -> bytes:
    frame = image_tensor[0:1] if image_tensor.dim() == 4 else image_tensor.unsqueeze(0)
    image = _tensor_to_pil(frame).convert("RGB")
    w, h = image.size
    long_side = max(w, h)
    if long_side > max_long_side and long_side > 0:
        scale = float(max_long_side) / float(long_side)
        new_w = max(16, int(round(w * scale)))
        new_h = max(16, int(round(h * scale)))
        resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image = image.resize((new_w, new_h), resampling)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str) and part.strip():
                parts.append(part.strip())
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text.strip()
    return ""


def _format_response_debug(response: requests.Response, raw_text: str, limit: int = 800) -> str:
    content_type = response.headers.get("content-type", "")
    body = (raw_text or "").strip()
    body = body[:limit] if body else "<empty>"
    return f"HTTP {response.status_code}, content-type={content_type or 'unknown'}, body={body}"


def _extract_text_from_chat_response_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        collected: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                text = _extract_text_from_content(message.get("content", ""))
                if text:
                    collected.append(text)
                    continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                text = _extract_text_from_content(delta.get("content", ""))
                if text:
                    collected.append(text)
                    continue
            text = choice.get("text")
            if isinstance(text, str) and text.strip():
                collected.append(text.strip())
        if collected:
            return "\n".join(collected).strip()

    text = _extract_text_from_content(payload.get("content", ""))
    if text:
        return text

    output = payload.get("output")
    if isinstance(output, list):
        collected = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_text = _extract_text_from_content(item.get("content", ""))
            if item_text:
                collected.append(item_text)
        if collected:
            return "\n".join(collected).strip()

    return ""


def _extract_text_from_chat_sse(raw_text: str) -> Tuple[str, str]:
    collected: List[str] = []
    last_error = ""
    saw_chunk = False
    for raw_line in (raw_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except Exception:
            last_error = line[:300]
            continue
        if isinstance(payload, dict):
            obj = str(payload.get("object", "") or "").strip().lower()
            if obj == "chat.completion.chunk":
                saw_chunk = True
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            last_error = json.dumps(payload.get("error"), ensure_ascii=False)[:800]
            continue
        text = _extract_text_from_chat_response_payload(payload)
        if text:
            collected.append(text)
    if not collected and not last_error and saw_chunk:
        last_error = "upstream returned stream chunks but no content text (empty delta)"
    return "\n".join(collected).strip(), last_error


def _extract_url_from_text(text: str) -> Optional[str]:
    m = re.search(r"https?://[^\s\"'<>]+", text or "")
    if not m:
        return None
    return m.group(0).rstrip(").,]}")


def _extract_urls_from_text(text: str, base_url: str = "") -> List[str]:
    raw = text or ""
    urls: List[str] = []
    seen: set[str] = set()

    def _push(u: str) -> None:
        cleaned = (u or "").strip().rstrip(").,]}")
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        urls.append(cleaned)

    for m in re.findall(r"!\[[^\]]*\]\((https?://[^\)]+)\)", raw):
        _push(m)
    for m in re.findall(r"https?://[^\s\"'<>]+", raw):
        _push(m)

    base = (base_url or "").rstrip("/")
    if base:
        for m in re.findall(r"(/v1/[^\s\"'<>]+)", raw):
            _push(base + m)

    return urls


def _extract_first_url(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return _extract_url_from_text(content)
    if isinstance(content, list):
        for part in content:
            url = _extract_first_url(part)
            if url:
                return url
        return None
    if isinstance(content, dict):
        image_url = content.get("image_url")
        if isinstance(image_url, dict):
            value = image_url.get("url")
            if isinstance(value, str) and value.startswith("http"):
                return value
        if isinstance(image_url, str) and image_url.startswith("http"):
            return image_url
        for key in ("url", "text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                url = _extract_url_from_text(value)
            else:
                url = _extract_first_url(value)
            if url:
                return url
    return None


def _looks_like_data_uri(value: str) -> bool:
    return isinstance(value, str) and value.startswith("data:image/") and ";base64," in value


def _collect_values(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            key_l = str(key).lower()
            if isinstance(value, str):
                if key_l in {
                    "url",
                    "imageurl",
                    "image_url",
                    "outputimageurl",
                    "output_image_url",
                    "b64_json",
                    "base64",
                    "imagebase64",
                    "image_base64",
                    "data",
                    "result",
                }:
                    yield value
                elif _looks_like_data_uri(value) or value.startswith(("http://", "https://")):
                    yield value
            yield from _collect_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _collect_values(item)
    elif isinstance(node, str):
        if _looks_like_data_uri(node) or node.startswith(("http://", "https://")):
            yield node


def _decode_maybe_base64(value: str) -> Optional[bytes]:
    text = (value or "").strip()
    if not text:
        return None
    if _looks_like_data_uri(text):
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return None


def _resize_image_batch(image_batch: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if image_batch.dim() != 4:
        return image_batch
    if image_batch.shape[1] == target_h and image_batch.shape[2] == target_w:
        return image_batch
    nchw = image_batch.permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.permute(0, 2, 3, 1)


def _stack_images(images: List[torch.Tensor]) -> torch.Tensor:
    if not images:
        return _blank_image()
    first = images[0] if images[0].dim() == 4 else images[0].unsqueeze(0)
    target_h, target_w = first.shape[1], first.shape[2]
    out: List[torch.Tensor] = []
    for img in images:
        batch = img if img.dim() == 4 else img.unsqueeze(0)
        out.append(_resize_image_batch(batch, target_h, target_w))
    return torch.cat(out, dim=0)


def _get_image_hw(image_tensor: torch.Tensor) -> Optional[Tuple[int, int]]:
    sample = image_tensor[0] if image_tensor.dim() == 4 else image_tensor
    if sample.dim() != 3:
        return None
    shape = tuple(sample.shape)
    if shape[-1] in (1, 3, 4):
        h, w = shape[0], shape[1]
    elif shape[0] in (1, 3, 4):
        h, w = shape[1], shape[2]
    else:
        h, w = shape[0], shape[1]
    if h <= 0 or w <= 0:
        return None
    return h, w


def _ratio_to_float(ratio: str) -> Optional[float]:
    if ":" not in (ratio or ""):
        return None
    left, right = ratio.split(":", 1)
    try:
        a = float(left)
        b = float(right)
    except ValueError:
        return None
    if a <= 0 or b <= 0:
        return None
    return a / b


def _choose_best_aspect_ratio(image_tensor: torch.Tensor) -> Optional[str]:
    hw = _get_image_hw(image_tensor)
    if hw is None:
        return None
    h, w = hw
    actual = float(w) / float(h)
    best_ratio: Optional[str] = None
    best_score = float("inf")
    for ratio_name in GEMINI_ASPECT_RATIO_OPTIONS:
        if ratio_name == AUTO_LABEL:
            continue
        ratio_val = _ratio_to_float(ratio_name)
        if ratio_val is None:
            continue
        score = abs(math.log(actual / ratio_val))
        if score < best_score:
            best_score = score
            best_ratio = ratio_name
    return best_ratio


def _normalize_aspect_ratio_label(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return AUTO_LABEL
    if raw.lower() == "auto":
        return AUTO_LABEL
    if raw == AUTO_LABEL:
        return AUTO_LABEL
    if raw in GEMINI_ASPECT_RATIO_OPTIONS:
        return raw
    return AUTO_LABEL


def _choose_best_aspect_ratio_from_options(image_tensor: torch.Tensor, ratio_options: List[str]) -> Optional[str]:
    hw = _get_image_hw(image_tensor)
    if hw is None:
        return None
    h, w = hw
    actual = float(w) / float(h)
    best_ratio: Optional[str] = None
    best_score = float("inf")
    for ratio_name in ratio_options:
        name = (ratio_name or "").strip()
        if not name or name.lower() == "auto":
            continue
        ratio_val = _ratio_to_float(name)
        if ratio_val is None:
            continue
        score = abs(math.log(actual / ratio_val))
        if score < best_score:
            best_score = score
            best_ratio = name
    return best_ratio


def _normalize_openai_aspect_ratio_label(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return "自动"
    if raw in {"auto", "自动"}:
        return "自动"
    normalized = raw.replace("x", ":")
    if normalized in OPENAI_IMAGE_ASPECT_RATIO_OPTIONS:
        return normalized
    return "自动"


def _ratio_colon_to_x(value: str) -> str:
    val = (value or "").strip().lower()
    if not val:
        return "1x1"
    if val in {"auto", "自动"}:
        return "1x1"
    return val.replace(":", "x")


def _build_openai_doc_model_name(base_model: str, ratio_colon: str) -> str:
    model = (base_model or "").strip()
    if not model:
        return model
    lower_model = model.lower()
    ratio_token = _ratio_colon_to_x(ratio_colon)

    # If already fully-qualified (includes ratio), keep as-is.
    if re.search(r"-(1k|2k|4k)-([0-9]+x[0-9]+)$", lower_model):
        return model

    # New channel: GPT-Image2-{1k|2k|4k} as preset; final model accepts ratio suffix.
    # Example: GPT-Image2-2k-2x3
    m = re.match(r"^(gpt-image2)-(1k|2k|4k)$", lower_model)
    if m:
        return f"{model}-{ratio_token}"
    if lower_model == "gpt-image2":
        return f"GPT-Image2-2k-{ratio_token}"

    return model


def _build_generate_content_url(base_url: str, model_name: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    model = (model_name or "").strip()
    if not base or not model:
        raise ValueError("invalid base url or model")
    if model.startswith("models/"):
        model = model.split("/", 1)[1]
    if base.endswith(":generateContent"):
        return base
    if "/models/" in base:
        return f"{base}:generateContent"
    if base.endswith("/v1"):
        root = base[:-3]
        return f"{root}/v1beta/models/{model}:generateContent"
    if base.endswith("/v1beta"):
        return f"{base}/models/{model}:generateContent"
    return f"{base}/v1beta/models/{model}:generateContent"


def _build_generate_content_urls(base_url: str, model_name: str) -> List[str]:
    base = (base_url or "").strip().rstrip("/")
    model = (model_name or "").strip()
    if not base or not model:
        return []
    if model.startswith("models/"):
        model = model.split("/", 1)[1]

    urls: List[str] = []
    if base.endswith("/v1"):
        root = base[:-3]
        urls.append(f"{root}/v1beta/models/{model}:generateContent")
        urls.append(f"{root}/v1/models/{model}:generateContent")
    elif base.endswith("/v1beta"):
        urls.append(f"{base}/models/{model}:generateContent")
        urls.append(f"{base[:-7]}/v1/models/{model}:generateContent")
    else:
        urls.append(f"{base}/v1beta/models/{model}:generateContent")
        urls.append(f"{base}/v1/models/{model}:generateContent")

    deduped: List[str] = []
    for url in urls:
        if url not in deduped:
            deduped.append(url)
    return deduped


def _post_json(url: str, api_key: str, payload: Dict[str, Any], timeout: Tuple[int, int]) -> requests.Response:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return _http_post(url, headers=_make_headers(api_key), data=body, timeout=timeout)


def _extract_image_tensor_from_response(data: Dict[str, Any]) -> Optional[torch.Tensor]:
    # OpenAI-compatible: choices[0].message.content has URL
    choices = data.get("choices", [])
    if isinstance(choices, list) and choices:
        item = choices[0] if isinstance(choices[0], dict) else {}
        message = item.get("message", {}) if isinstance(item, dict) else {}
        content = message.get("content", "")
        image_url = _extract_first_url(content)
        if image_url:
            return _download_image_to_tensor(image_url)

    # Gemini-native: candidates[].content.parts[]
    candidates = data.get("candidates", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict):
                    b64 = inline.get("data")
                    if isinstance(b64, str) and b64:
                        tensor = _decode_base64_image_to_tensor(b64)
                        if tensor is not None:
                            return tensor
                file_data = part.get("fileData") or part.get("file_data")
                if isinstance(file_data, dict):
                    file_url = file_data.get("fileUri") or file_data.get("uri") or file_data.get("url")
                    if isinstance(file_url, str) and file_url.startswith("http"):
                        return _download_image_to_tensor(file_url)
                text_val = part.get("text")
                if isinstance(text_val, str):
                    text_url = _extract_url_from_text(text_val)
                    if text_url:
                        return _download_image_to_tensor(text_url)
    return None


def _image_task_query_url(task_id: str) -> str:
    return f"{OPENAI_API_V1}/images/tasks/{str(task_id or '').strip()}"


def _extract_native_image_task_info(payload: Any) -> Tuple[str, str]:
    task_id = ""
    if isinstance(payload, dict):
        for key in ("task_id", "id", "taskId"):
            value = payload.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                task_id = str(value).strip()
                break

        if not task_id:
            data = payload.get("data")
            if isinstance(data, dict):
                for key in ("task_id", "id", "taskId"):
                    value = data.get(key)
                    if isinstance(value, (str, int)) and str(value).strip():
                        task_id = str(value).strip()
                        break

    return task_id, _image_task_query_url(task_id) if task_id else ""


def _create_reference_gemini_request(
    prompt: str,
    seed: int,
    aspect_ratio: str,
    image_size: str,
    input_images_b64: List[str],
    include_image_size: bool = True,
    add_prompt_config_suffix: bool = True,
    include_top_p: bool = True,
) -> Dict[str, Any]:
    prompt_text = (prompt or "").strip()
    if not prompt_text and not input_images_b64:
        raise ValueError("prompt or reference image is required")

    # Some upstream routes require at least one user text part in contents.
    # Keep image-only workflows working by injecting a minimal fallback text.
    if not prompt_text:
        prompt_text = "请根据参考图生成图像"

    normalized_ratio = (aspect_ratio or "").strip()
    normalized_size = (image_size or "").strip().upper()
    if add_prompt_config_suffix:
        suffix_parts: List[str] = []
        if include_image_size and normalized_size in {"1K", "2K", "4K"}:
            suffix_parts.append(f"\u5206\u8fa8\u7387: {normalized_size}")

        if normalized_ratio and normalized_ratio.lower() != "auto":
            suffix_parts.append(f"\u6bd4\u4f8b: {normalized_ratio}")

        if prompt_text and suffix_parts:
            prompt_text = prompt_text + " [" + ", ".join(suffix_parts) + "]"

    parts: List[Dict[str, Any]] = []
    if prompt_text:
        parts.append({"text": prompt_text})

    for encoded in input_images_b64:
        if not encoded:
            continue
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": encoded,
                }
            }
        )

    generation_config: Dict[str, Any] = {
        "responseModalities": ["IMAGE"],
    }
    if include_top_p:
        generation_config["topP"] = 0.95
    if isinstance(seed, int) and seed >= 0:
        generation_config["seed"] = seed

    image_config: Dict[str, Any] = {}
    if normalized_ratio and normalized_ratio.lower() != "auto":
        image_config["aspectRatio"] = normalized_ratio
    if include_image_size and normalized_size in {"1K", "2K", "4K"}:
        image_config["imageSize"] = normalized_size
    if image_config:
        generation_config["imageConfig"] = image_config

    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }


def _extract_reference_gemini_images(response_data: Dict[str, Any]) -> Tuple[List[str], str]:
    images: List[str] = []
    source_types: List[str] = []
    candidates = response_data.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue

            inline = part.get("inlineData")
            if isinstance(inline, dict):
                data = inline.get("data")
                mime = inline.get("mimeType", "")
                if isinstance(data, str) and data and str(mime).startswith("image/"):
                    images.append(data)
                    source_types.append("base64")
                    continue

            file_data = part.get("fileData")
            if isinstance(file_data, dict):
                file_uri = file_data.get("fileUri") or file_data.get("uri") or file_data.get("url")
                if isinstance(file_uri, str) and file_uri.startswith("http"):
                    downloaded = _download_image_to_base64(file_uri)
                    if downloaded:
                        images.append(downloaded)
                        source_types.append("url")
                        continue

            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                text_url = _extract_url_from_text(text_value.strip())
                if text_url:
                    downloaded = _download_image_to_base64(text_url)
                    if downloaded:
                        images.append(downloaded)
                        source_types.append("url")
                        continue
    if not source_types:
        source = "unknown"
    elif all(t == source_types[0] for t in source_types):
        source = source_types[0]
    else:
        source = "mixed"
    return images, source


def _gemini_should_include_image_size(model_name: str) -> bool:
    text = (model_name or "").strip().lower()
    text = re.sub(r"^\s*[【\[]\s*[a-z]\s*[】\]]\s*", "", text).strip()
    # Normalize optional ratio suffix, e.g. nano-banana2-1k-1x1 -> nano-banana2-1k
    text = re.sub(r"-\d+x\d+$", "", text)
    stream_exact_models = {
        "nano-banana2-1k",
        "nano-banana2-2k",
        "nano-banana2-4k",
        "nano-banana-pro-1k",
        "nano-banana-pro-2k",
        "nano-banana-pro-4k",
    }
    return text not in stream_exact_models


def _gemini_should_map_ratio_to_x(model_name: str) -> bool:
    text = (model_name or "").strip().lower()
    # Strip optional UI prefix markers like 【X】 / [X]
    text = re.sub(r"^\s*[【\[]\s*[a-z]\s*[】\]]\s*", "", text).strip()
    # Normalize optional ratio suffix, e.g. nano-banana2-1k-1x1 -> nano-banana2-1k
    text = re.sub(r"-\d+x\d+$", "", text)

    # Only these six model names map ratio like 9:16 -> 9x16.
    stream_exact_models = {
        "nano-banana2-1k",
        "nano-banana2-2k",
        "nano-banana2-4k",
        "nano-banana-pro-1k",
        "nano-banana-pro-2k",
        "nano-banana-pro-4k",
    }
    return text in stream_exact_models


def _gemini_ratio_for_model(model_name: str, ratio_value: str) -> str:
    ratio = (ratio_value or "").strip()
    if not ratio:
        return ratio
    if _gemini_should_map_ratio_to_x(model_name):
        if ratio.lower() != "auto":
            return ratio.replace(":", "x")
    return ratio


def _set_gemini_async_task_state(task: Dict[str, Any], **updates: Any) -> None:
    with GEMINI_ASYNC_QUEUE_LOCK:
        task.update(updates)


def _prepare_gemini_image_task_from_kwargs(kwargs: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    prompt = kwargs.get("prompt", kwargs.get("提示词", ""))
    model_preset = kwargs.get("模型预设", kwargs.get("model_preset", "【X】gemini-3.1-flash-image-preview"))
    custom_model = kwargs.get("自定义模型", kwargs.get("custom_model", ""))
    api_key = kwargs.get("API密钥", kwargs.get("api_key", ""))
    image_size = kwargs.get("图像尺寸", kwargs.get("image_size", "4K"))
    aspect_ratio = kwargs.get("图像比例", kwargs.get("aspect_ratio", AUTO_LABEL))
    seed = int(kwargs.get("种子", kwargs.get("seed", 0)))

    if not (api_key or "").strip():
        return None, "api_key is required"

    model_name = _resolve_model_name(model_preset, custom_model)
    if not (model_name or "").strip():
        return None, "model name is empty"
    include_image_size = _gemini_should_include_image_size(model_name)

    normalized_ratio = _normalize_aspect_ratio_label(aspect_ratio)
    resolved_ratio = normalized_ratio
    if normalized_ratio == AUTO_LABEL:
        img1 = kwargs.get("参考图1", kwargs.get("image_1"))
        if isinstance(img1, torch.Tensor):
            best_ratio = _choose_best_aspect_ratio(img1)
            if best_ratio:
                resolved_ratio = best_ratio

    ratio_for_api = "Auto" if resolved_ratio == AUTO_LABEL else resolved_ratio
    ratio_for_api = _gemini_ratio_for_model(model_name, ratio_for_api)
    size_for_api = (image_size or "2K").strip().upper()
    if size_for_api not in {"1K", "2K", "4K"}:
        size_for_api = "2K"

    seed_int = int(seed)
    seed_for_api = seed_int % 2147483647 if seed_int > 0 else -1

    input_images_b64: List[str] = []
    ref_count = 0
    for i in range(1, 15):
        key_cn = f"参考图{i}"
        key_en = f"image_{i}"
        image_input = kwargs.get(key_cn, kwargs.get(key_en))
        if not isinstance(image_input, torch.Tensor):
            continue
        try:
            if image_input.dim() == 4:
                for idx in range(image_input.shape[0]):
                    input_images_b64.append(_tensor_to_base64(image_input[idx : idx + 1]))
                    ref_count += 1
            else:
                input_images_b64.append(_tensor_to_base64(image_input))
                ref_count += 1
        except Exception as exc:
            _log(f"Gemini image encode failed: {key_cn} - {exc}")

    request_data = _create_reference_gemini_request(
        prompt=prompt,
        seed=seed_for_api,
        aspect_ratio=ratio_for_api,
        image_size=size_for_api,
        input_images_b64=input_images_b64,
        include_image_size=include_image_size,
    )

    task_payload = {
        "prompt": prompt,
        "api_key": (api_key or "").strip(),
        "model_name": model_name,
        "size_for_api": size_for_api,
        "ratio_for_api": ratio_for_api,
        "seed_for_api": seed_for_api,
        "ref_count": ref_count,
        "include_image_size": include_image_size,
        "request_data": request_data,
    }
    return task_payload, ""


def _build_gemini_stream_message_content_from_request(request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    msg_content: List[Dict[str, Any]] = []
    contents = request_data.get("contents") if isinstance(request_data, dict) else None
    if isinstance(contents, list) and contents and isinstance(contents[0], dict):
        parts = contents[0].get("parts")
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text_val = part.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    msg_content.append({"type": "text", "text": text_val.strip()})
                    continue
                inline = part.get("inlineData")
                if isinstance(inline, dict):
                    b64 = inline.get("data")
                    mime = str(inline.get("mimeType", "image/png") or "image/png")
                    if isinstance(b64, str) and b64.strip():
                        msg_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64.strip()}"},
                            }
                        )
    return msg_content


def _extract_openai_compatible_images_to_base64(data: Dict[str, Any]) -> List[str]:
    images: List[str] = []
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            urls = _extract_urls_from_text(_stringify_content(content), OPENAI_API_ROOT.rstrip("/"))
            for url in urls:
                downloaded = _download_image_to_base64(url)
                if downloaded:
                    images.append(downloaded)
    return images


def _execute_gemini_stream_compat_task(task_payload: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], str, str]:
    model_name = task_payload.get("model_name", "")
    api_key = task_payload.get("api_key", "")
    request_data = task_payload.get("request_data", {}) or {}
    ratio_for_api = str(task_payload.get("ratio_for_api", "") or "")
    seed_for_api = int(task_payload.get("seed_for_api", -1))

    msg_content = _build_gemini_stream_message_content_from_request(request_data)
    if not msg_content:
        msg_content = [{"type": "text", "text": " "}]

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": msg_content}],
        "stream": True,
        "n": 1,
    }
    if ratio_for_api and ratio_for_api.lower() != "auto":
        ratio_x = ratio_for_api.replace(":", "x")
        payload["size"] = ratio_x
        payload["aspect_ratio"] = ratio_x
    if seed_for_api >= 0:
        payload["seed"] = seed_for_api

    headers = {
        "Authorization": f"Bearer {(api_key or '').strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    try:
        response = _http_post(
            CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(30, 300),
        )
    except Exception as exc:
        return None, "unknown", f"chat(stream) exception: {exc}"

    if response.status_code != 200:
        return None, "unknown", f"chat(stream) HTTP {response.status_code}: {(response.text or '')[:300]}"

    content_type = str(response.headers.get("Content-Type", "")).lower()
    images_b64: List[str] = []
    result_format = "unknown"

    if "text/event-stream" in content_type:
        stream_images: List[str] = []
        stream_source = "unknown"
        stream_err = ""
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = (raw or "").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str:
                continue
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except Exception:
                continue

            if isinstance(chunk, dict) and isinstance(chunk.get("error"), dict):
                err_obj = chunk.get("error", {})
                stream_err = str(err_obj.get("message", "")) or json.dumps(err_obj, ensure_ascii=False)
                continue

            # Path A: Gemini-native chunk fields (candidates.parts.inlineData)
            part_images, part_source = _extract_reference_gemini_images(chunk if isinstance(chunk, dict) else {})
            if part_images:
                stream_images.extend(part_images)
                if stream_source == "unknown":
                    stream_source = part_source
                continue

            # Path B: OpenAI-like stream chunk content with image URL.
            content = ""
            if isinstance(chunk, dict):
                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    delta = choices[0].get("delta", {})
                    if isinstance(delta, dict):
                        val = delta.get("content", "")
                        if isinstance(val, str):
                            content = val
            if content:
                urls = _extract_urls_from_text(content, OPENAI_API_ROOT.rstrip("/"))
                for u in urls:
                    downloaded = _download_image_to_base64(u)
                    if downloaded:
                        stream_images.append(downloaded)
                        stream_source = "url"

        if stream_images:
            images_b64, result_format = stream_images, stream_source
        elif stream_err:
            return None, "unknown", f"chat(stream) error: {stream_err}"
        else:
            return None, "unknown", "chat(stream) has no image"
    else:
        try:
            data = response.json()
        except Exception:
            return None, "unknown", "chat(stream) non-json response"

        images_b64, result_format = _extract_reference_gemini_images(data)
        if not images_b64:
            images_b64 = _extract_openai_compatible_images_to_base64(data)
            if images_b64:
                result_format = "url"
        if not images_b64:
            return None, "unknown", f"chat(stream) no image in response: {json.dumps(data, ensure_ascii=False)[:300]}"

    tensor = _decode_base64_image_to_tensor(images_b64[0]) if images_b64 else None
    if tensor is None:
        return None, "unknown", "chat(stream) returned image data but decode failed"
    return tensor, result_format, ""


def _extract_async_task_info_from_chat_response(payload: Any) -> Tuple[str, str]:
    task_id = ""
    query_url = ""
    if not isinstance(payload, dict):
        return task_id, query_url

    task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
    query_url = str(payload.get("query_url") or "").strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message") if isinstance(first, dict) else {}
        content = msg.get("content") if isinstance(msg, dict) else ""
        parsed: Any = None
        if isinstance(content, str):
            text = content.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
        elif isinstance(content, dict):
            parsed = content
        if isinstance(parsed, dict):
            task_id = str(parsed.get("task_id") or parsed.get("id") or task_id).strip()
            query_url = str(parsed.get("query_url") or query_url).strip()

    if task_id and not query_url:
        query_url = _image_task_query_url(task_id)
    return task_id, query_url


def _extract_async_image_result_urls(payload: Any) -> List[str]:
    urls: List[str] = []

    def scan(node: Any, key: str = "", depth: int = 0) -> None:
        if depth > 12 or node is None:
            return
        if isinstance(node, dict):
            for child_key, value in node.items():
                scan(value, str(child_key), depth + 1)
            return
        if isinstance(node, list):
            for item in node:
                scan(item, key, depth + 1)
            return
        if not isinstance(node, str):
            return

        text = node.strip()
        if not text:
            return
        if key in {"url", "image_url", "content", "text", "fileUri", "file_uri"}:
            if text.startswith(("http://", "https://")):
                urls.append(text)
            urls.extend(_extract_urls_from_text(text, OPENAI_API_ROOT.rstrip("/")))

    scan(payload)
    dedup: List[str] = []
    seen: set[str] = set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


def _poll_gemini_async_task(
    api_key: str,
    task_id: str,
    query_url: str = "",
    max_attempts: int = 300,
    interval_sec: float = 3.0,
) -> Dict[str, Any]:
    q_url = (query_url or "").strip() or _image_task_query_url(task_id)
    headers = {"Authorization": f"Bearer {(api_key or '').strip()}"}
    last_payload: Dict[str, Any] = {}

    for attempt in range(1, max_attempts + 1):
        try:
            response = _http_get(q_url, headers=headers, timeout=(20, 120))
            if response.status_code != 200:
                _log(
                    f"Gemini task poll failed: attempt={attempt}, "
                    f"http={response.status_code}, body={(response.text or '')[:220]}"
                )
                time.sleep(interval_sec)
                continue
            payload = response.json()
        except Exception as exc:
            _log(f"Gemini task poll exception: attempt={attempt}, error={exc}")
            time.sleep(interval_sec)
            continue

        if isinstance(payload, dict):
            last_payload = payload
        status = str((payload or {}).get("status", "")).strip().lower()
        has_url = bool(_extract_async_image_result_urls(payload))
        _log(
            f"Gemini task poll: attempt={attempt}, status={status or 'unknown'}, "
            f"has_url={'yes' if has_url else 'no'}"
        )

        if status in {"completed", "done", "success", "succeeded"}:
            return payload if isinstance(payload, dict) else {}
        if status in {"failed", "error", "canceled", "cancelled"}:
            return payload if isinstance(payload, dict) else {}

        time.sleep(interval_sec)

    return last_payload


def _submit_gemini_task_to_gateway(task_payload: Dict[str, Any]) -> Tuple[str, str, Optional[torch.Tensor], str]:
    """Submit Gemini image requests through the native generateContent async route."""
    model_name = str(task_payload.get("model_name", "") or "").strip()
    api_key = str(task_payload.get("api_key", "") or "").strip()

    if not model_name or not api_key:
        return "", "", None, "missing model or api_key"

    request_data = task_payload.get("request_data") or {}
    if not isinstance(request_data, dict) or not request_data.get("contents"):
        return "", "", None, "request_data is empty or invalid"

    target_url = _build_generate_content_url(BASE_URL, model_name)
    _log(
        f"Gemini native async submit: model={model_name}, "
        f"ratio={task_payload.get('ratio_for_api')}, size={task_payload.get('size_for_api')}"
    )

    try:
        response = _http_post(
            target_url,
            headers=_make_headers(api_key),
            json=request_data,
            timeout=(120, 600),
        )
    except Exception as exc:
        return "", "", None, f"native async submit exception: {exc}"

    if response.status_code not in (200, 201, 202):
        return "", "", None, f"native async submit HTTP {response.status_code}: {(response.text or '')[:400]}"

    try:
        payload = response.json()
    except Exception:
        return "", "", None, f"native async submit non-json: {(response.text or '')[:400]}"

    task_id, query_url = _extract_native_image_task_info(payload)
    if not task_id and not query_url:
        tensor = _extract_image_tensor_from_response(payload)
        if tensor is not None:
            return "", "", tensor, ""
        return "", "", None, f"native submit response has no task info: {json.dumps(payload, ensure_ascii=False)[:400]}"

    _log(
        f"Gemini native async task created: {task_id or 'unknown'}, "
        f"query_url={query_url or 'empty'}"
    )
    return task_id, query_url, None, ""


def _finalize_gemini_task_from_query(api_key: str, task_id: str, query_url: str, max_attempts: int = 200) -> Tuple[Optional[torch.Tensor], str]:
    """轮询 + 下载结果。可在子线程或主线程调用。
    返回值：(tensor or None, error_message)
    """
    final_payload = _poll_gemini_async_task(api_key, task_id, query_url=query_url, max_attempts=max_attempts, interval_sec=3.0)
    final_status = str((final_payload or {}).get("status", "")).strip().lower()
    result_urls = _extract_async_image_result_urls(final_payload)

    if final_status in {"completed", "done", "success", "succeeded"} and result_urls:
        try:
            tensor = _download_image_to_tensor(result_urls[0])
            return tensor, ""
        except Exception as exc:
            return None, f"download image failed: {exc}"
    if final_status in {"failed", "error", "canceled", "cancelled"}:
        return None, f"async task failed: {json.dumps(final_payload, ensure_ascii=False)[:400]}"
    return None, f"async task no image: {json.dumps(final_payload, ensure_ascii=False)[:400]}"


def _execute_gemini_via_async_gateway(task_payload: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], str, str]:
    """同步路径用：提交 + 轮询 + 下载在同一线程里串行完成。"""
    api_key = str(task_payload.get("api_key", "") or "").strip()
    task_id, query_url, direct_tensor, error = _submit_gemini_task_to_gateway(task_payload)
    if error:
        return None, "unknown", error
    if direct_tensor is not None:
        return direct_tensor, "url", ""

    tensor, err = _finalize_gemini_task_from_query(api_key, task_id, query_url)
    if tensor is None:
        return None, "unknown", err
    return tensor, "url", ""


def _build_openai_chat_messages_from_request_data(request_data: Dict[str, Any], prompt: str) -> List[Dict[str, Any]]:
    """把 _create_reference_gemini_request 产生的 contents.parts 还原成 OpenAI chat messages 结构。
    parts 里的 inlineData -> image_url(data: 协议 base64);text -> text part。
    """
    content: List[Dict[str, Any]] = []

    contents = request_data.get("contents") if isinstance(request_data, dict) else None
    if isinstance(contents, list) and contents and isinstance(contents[0], dict):
        parts = contents[0].get("parts")
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict):
                    b64 = inline.get("data")
                    mime = str(inline.get("mimeType") or inline.get("mime_type") or "image/png")
                    if isinstance(b64, str) and b64.strip():
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64.strip()}"},
                        })
                        continue
                text_value = part.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    content.append({"type": "text", "text": text_value.strip()})

    if not content:
        # 兜底：至少有一个文本 part
        content.append({"type": "text", "text": (prompt or "").strip() or "请生成图像"})

    return [{"role": "user", "content": content}]


def _execute_gemini_native_sync(task_payload: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], str, str]:
    """KR-Gemini sync node: submit a native async task, then poll it locally."""
    return _execute_gemini_via_async_gateway(task_payload)


def _execute_gemini_image_task(task_payload: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], str, str]:
    # KR-Gemini生图(同步) 也强制走 Gemini 原生异步任务，再在节点内部轮询拿图。
    tensor, result_format, error_message = _execute_gemini_native_sync(task_payload)
    if tensor is not None:
        return tensor, result_format, ""
    return None, "unknown", error_message or "unknown error"


def _gemini_async_worker_loop(worker_name: str) -> None:
    while True:
        task: Optional[Dict[str, Any]] = None
        with GEMINI_ASYNC_QUEUE_LOCK:
            if GEMINI_ASYNC_SUBMISSION_QUEUE:
                task = GEMINI_ASYNC_SUBMISSION_QUEUE.pop(0)
        if task is None:
            time.sleep(0.1)
            continue

        try:
            _set_gemini_async_task_state(task, status="RUNNING", started_at=time.time())
            payload = task.get("task_payload") or {}
            _set_bypass_proxy(bool(task.get("bypass_proxy", payload.get("绕过代理", True))))
            api_key = str(payload.get("api_key", "") or "").strip()
            upstream_task_id = str(task.get("upstream_task_id", "") or "")
            query_url = str(task.get("query_url", "") or "")
            _log(
                f"Gemini async[{worker_name}] start polling: id={task.get('task_id')}, "
                f"upstream={upstream_task_id}, model={payload.get('model_name')}, "
                f"refs={payload.get('ref_count')}"
            )

            # 异步路径已经在主线程提交完成，子线程只负责轮询 + 下载结果。
            # 这样大 body POST 不会落在子线程里碰到 ProxyError。
            if not upstream_task_id and not query_url:
                # 极少数情况：提交直接返图（没有 task_id），主线程已把 tensor 塞在 task 里
                pre_tensor = task.get("result_tensor")
                if isinstance(pre_tensor, torch.Tensor):
                    _set_gemini_async_task_state(
                        task,
                        status="DONE",
                        result_format="url",
                        error="",
                        finished_at=time.time(),
                    )
                    _log(f"Gemini async[{worker_name}] done(direct): id={task.get('task_id')}")
                    continue
                _set_gemini_async_task_state(
                    task,
                    status="FAILED",
                    result_tensor=None,
                    result_format="unknown",
                    error="missing upstream task_id and query_url",
                    finished_at=time.time(),
                )
                _log(f"Gemini async[{worker_name}] failed: id={task.get('task_id')}, missing upstream task info")
                continue

            tensor, error_message = _finalize_gemini_task_from_query(api_key, upstream_task_id, query_url)
            if tensor is not None:
                _set_gemini_async_task_state(
                    task,
                    status="DONE",
                    result_tensor=tensor,
                    result_format="url",
                    error="",
                    finished_at=time.time(),
                )
                _log(f"Gemini async[{worker_name}] done: id={task.get('task_id')}")
            else:
                _set_gemini_async_task_state(
                    task,
                    status="FAILED",
                    result_tensor=None,
                    result_format="unknown",
                    error=error_message,
                    finished_at=time.time(),
                )
                _log(f"Gemini async[{worker_name}] failed: id={task.get('task_id')}, error={error_message}")
        except Exception as exc:
            _set_gemini_async_task_state(
                task,
                status="FAILED",
                result_tensor=None,
                result_format="unknown",
                error=f"worker exception: {exc}",
                finished_at=time.time(),
            )
            _log(f"Gemini async[{worker_name}] exception: {exc}")


def _ensure_gemini_async_workers_started() -> None:
    global GEMINI_ASYNC_WORKERS_STARTED
    if GEMINI_ASYNC_WORKERS_STARTED:
        return
    with GEMINI_ASYNC_QUEUE_LOCK:
        if GEMINI_ASYNC_WORKERS_STARTED:
            return
        for i in range(GEMINI_ASYNC_WORKER_COUNT):
            thread = threading.Thread(
                target=_gemini_async_worker_loop,
                args=(f"W{i + 1}",),
                daemon=True,
            )
            thread.start()
        GEMINI_ASYNC_WORKERS_STARTED = True
        _log(f"Gemini async workers started: {GEMINI_ASYNC_WORKER_COUNT}")


class KRLLMNode:
    @classmethod
    def INPUT_TYPES(cls):
        optional_images = {f"参考图{i}": ("IMAGE",) for i in range(1, 5)}
        return {
            "required": {
                "系统提示词": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "用户提示词": ("STRING", {"multiline": True, "default": ""}),
                "模型预设": (LLM_MODEL_PRESETS, {"default": "【R】gemini-3-pro-preview"}),
                "自定义模型": ("STRING", {"multiline": False, "default": ""}),
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "绕过代理": ("BOOLEAN", {"default": True}),
            },
            "optional": optional_images,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("\u6587\u672c",)
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME
    OUTPUT_NODE = True

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        system_prompt = kwargs.get("系统提示词", kwargs.get("system_prompt", "You are a helpful assistant."))
        user_prompt = kwargs.get("用户提示词", kwargs.get("user_prompt", ""))
        model_preset = kwargs.get("模型预设", kwargs.get("model_preset", "【R】gemini-3-pro-preview"))
        custom_model = kwargs.get("自定义模型", kwargs.get("custom_model", ""))
        api_key = kwargs.get("API密钥", kwargs.get("api_key", ""))
        if not (api_key or "").strip():
            return ("[Comfyui-Kr-API] api_key is required.",)

        content_parts: List[Dict[str, Any]] = []
        if (user_prompt or "").strip():
            content_parts.append({"type": "text", "text": user_prompt})

        ref_count = 0
        for i in range(1, 5):
            key_cn = f"参考图{i}"
            key_en = f"image_{i}"
            image_input = kwargs.get(key_cn, kwargs.get(key_en))
            if not isinstance(image_input, torch.Tensor):
                continue
            try:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _tensor_to_compressed_jpeg_data_url(image_input, max_long_side=1024, quality=100)},
                    }
                )
                ref_count += 1
            except Exception as exc:
                _log(f"LLM image encode failed: {key_cn} - {exc}")

        if not content_parts:
            return ("[Comfyui-Kr-API] user_prompt or at least one reference image is required.",)

        user_content: Any = content_parts if ref_count > 0 else user_prompt
        model_name = _resolve_model_name(model_preset, custom_model)
        _log(f"LLM request: model={model_name}, refs={ref_count}")

        payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        try:
            response = _http_post(CHAT_COMPLETIONS_URL, headers=_make_headers(api_key), json=payload, timeout=(30, 180))
            if response.status_code != 200:
                return (f"[Comfyui-Kr-API] request failed: HTTP {response.status_code} - {response.text[:300]}",)
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return (json.dumps(data, ensure_ascii=False),)
            message = choices[0].get("message", {})
            content = message.get("content", "")
            text = _extract_text_from_content(content)
            return (text if text else json.dumps(message, ensure_ascii=False),)
        except Exception as exc:
            return (f"[Comfyui-Kr-API] request exception: {exc}",)


class KRGPTLanguageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "系统提示词": ("STRING", {"multiline": True, "default": "You are a helpful assistant."}),
                "规则填写": ("STRING", {"multiline": True, "default": ""}),
                "自定义模型": ("STRING", {"multiline": False, "default": "gpt-4o"}),
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "绕过代理": ("BOOLEAN", {"default": True}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
            },
            "optional": {
                "图片": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("\u6587\u672c",)
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME
    OUTPUT_NODE = True

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        system_prompt = str(kwargs.get("系统提示词", kwargs.get("system_prompt", "")) or "").strip()
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
        rules = ""
        for key in ("规则填写", "rules", "prompt", "提示词", "用户提示词", "user_prompt"):
            value = kwargs.get(key)
            if isinstance(value, str) and value.strip():
                rules = value.strip()
                break
        model_name = str(kwargs.get("自定义模型", kwargs.get("custom_model", "gpt-4o")) or "").strip()
        api_key = kwargs.get("API密钥", kwargs.get("api_key", ""))
        seed = int(kwargs.get("种子", kwargs.get("seed", 0)))
        image_input = kwargs.get("图片", kwargs.get("image"))

        if not (api_key or "").strip():
            return ("[Comfyui-Kr-API] api_key is required.",)
        if not model_name:
            return ("[Comfyui-Kr-API] custom model is required.",)

        has_image = isinstance(image_input, torch.Tensor)
        rules_text = str(rules or "").strip()
        if not rules_text and not has_image:
            return ("[Comfyui-Kr-API] 规则填写 or 图片 is required.",)

        user_content: Any
        if has_image:
            content_parts: List[Dict[str, Any]] = []
            if rules_text:
                content_parts.append({"type": "text", "text": rules_text})
            else:
                content_parts.append({"type": "text", "text": "请根据图片生成文本。"})
            try:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _tensor_to_compressed_jpeg_data_url(image_input, max_long_side=1024, quality=100),
                            "detail": "low",
                        },
                    }
                )
            except Exception as exc:
                _log(f"GPT language image encode failed: {exc}")
                return (f"[Comfyui-Kr-API] image encode failed: {exc}",)
            user_content = content_parts
        else:
            user_content = rules_text

        payload: Dict[str, Any] = {
            "model": model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if seed > 0:
            payload["seed"] = seed % 2147483647

        _log(f"GPT language request: model={model_name}, image={1 if has_image else 0}, seed={payload.get('seed', 'none')}")

        try:
            response = _kr_post_json_with_fallback(
                CHAT_COMPLETIONS_URL,
                headers=_make_openai_compat_headers(api_key),
                payload=payload,
                timeout=(120, 300),
            )
            raw_text = response.text or ""
            if response.status_code != 200:
                return (f"[Comfyui-Kr-API] request failed: {_format_response_debug(response, raw_text)}",)
            if not raw_text.strip():
                return (f"[Comfyui-Kr-API] request failed: {_format_response_debug(response, raw_text)}",)

            content_type = response.headers.get("content-type", "").lower()
            looks_like_sse = (
                "text/event-stream" in content_type
                or raw_text.lstrip().startswith("data:")
                or "\ndata:" in raw_text
                or "data: [DONE]" in raw_text
                or '"object":"chat.completion.chunk"' in raw_text
            )
            if looks_like_sse:
                stream_text, stream_error = _extract_text_from_chat_sse(raw_text)
                if stream_text:
                    return (stream_text,)

                # Some OpenAI-compatible upstreams send SSE wrappers with empty delta content
                # in non-stream mode. Retry once with an explicit non-stream payload.
                _log(
                    "KRLLMNode stream-like response has no text, retrying with explicit non-stream. "
                    f"err={stream_error or raw_text[:220]}"
                )
                payload_retry = dict(payload)
                payload_retry["stream"] = False
                response_retry = _kr_post_json_with_fallback(
                    CHAT_COMPLETIONS_URL,
                    headers=_make_openai_compat_headers(api_key),
                    payload=payload_retry,
                    timeout=(120, 300),
                )
                retry_text = response_retry.text or ""
                if response_retry.status_code != 200:
                    return (
                        "[Comfyui-Kr-API] stream parse failed and non-stream retry failed: "
                        + _format_response_debug(response_retry, retry_text)
                    ,)
                try:
                    retry_data = response_retry.json()
                    parsed_text = _extract_text_from_chat_response_payload(retry_data)
                    if parsed_text:
                        return (parsed_text,)
                    return (json.dumps(retry_data, ensure_ascii=False),)
                except Exception:
                    if "data:" in retry_text or "[DONE]" in retry_text or "chat.completion.chunk" in retry_text:
                        retry_stream_text, retry_stream_error = _extract_text_from_chat_sse(retry_text)
                        if retry_stream_text:
                            return (retry_stream_text,)
                    return (
                        f"[Comfyui-Kr-API] stream response has no text (retry): "
                        f"{retry_stream_error if 'retry_stream_error' in locals() else _format_response_debug(response_retry, retry_text)}",
                    )

            try:
                data = response.json()
            except ValueError:
                # Defensive fallback: some upstreams return SSE body with wrong content-type.
                if "data:" in raw_text or "[DONE]" in raw_text or "chat.completion.chunk" in raw_text:
                    stream_text, stream_error = _extract_text_from_chat_sse(raw_text)
                    if stream_text:
                        return (stream_text,)
                    return (f"[Comfyui-Kr-API] stream response has no text: {stream_error or raw_text[:800]}",)
                return (raw_text.strip() if raw_text.strip() else f"[Comfyui-Kr-API] non-json response: {_format_response_debug(response, raw_text)}",)

            text = _extract_text_from_chat_response_payload(data)
            if text:
                return (text,)
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                return (f"[Comfyui-Kr-API] request failed: {json.dumps(data.get('error'), ensure_ascii=False)}",)
            return (json.dumps(data, ensure_ascii=False),)
        except Exception as exc:
            return (f"[Comfyui-Kr-API] request exception: {exc}",)


class KRGeminiImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        optional_images = {f"参考图{i}": ("IMAGE",) for i in range(1, 15)}
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "模型预设": (GEMINI_IMAGE_MODEL_PRESETS, {"default": "【X】gemini-3.1-flash-image-preview"}),
                "自定义模型": ("STRING", {"multiline": False, "default": ""}),
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "图像尺寸": (GEMINI_IMAGE_SIZE_PRESETS, {"default": "4K"}),
                "图像比例": (GEMINI_ASPECT_RATIO_OPTIONS, {"default": AUTO_LABEL}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "绕过代理": ("BOOLEAN", {"default": True}),
            },
            "optional": optional_images,
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("\u56fe\u50cf", "结果格式")
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        prompt = kwargs.get("prompt", kwargs.get("提示词", ""))
        model_preset = kwargs.get("模型预设", kwargs.get("model_preset", "【X】gemini-3.1-flash-image-preview"))
        custom_model = kwargs.get("自定义模型", kwargs.get("custom_model", ""))
        api_key = kwargs.get("API密钥", kwargs.get("api_key", ""))
        image_size = kwargs.get("图像尺寸", kwargs.get("image_size", "4K"))
        aspect_ratio = kwargs.get("图像比例", kwargs.get("aspect_ratio", AUTO_LABEL))
        seed = int(kwargs.get("种子", kwargs.get("seed", 0)))

        if not (api_key or "").strip():
            _log("Gemini node missing api_key.")
            return (_blank_image(), "unknown")

        model_name = _resolve_model_name(model_preset, custom_model)
        include_image_size = _gemini_should_include_image_size(model_name)
        normalized_ratio = _normalize_aspect_ratio_label(aspect_ratio)
        resolved_ratio = normalized_ratio
        if normalized_ratio == AUTO_LABEL:
            img1 = kwargs.get("参考图1", kwargs.get("image_1"))
            if isinstance(img1, torch.Tensor):
                best_ratio = _choose_best_aspect_ratio(img1)
                if best_ratio:
                    resolved_ratio = best_ratio
                    _log(f"Gemini auto aspect from reference image: {resolved_ratio}")

        ratio_for_api = "Auto" if resolved_ratio == AUTO_LABEL else resolved_ratio
        ratio_for_api = _gemini_ratio_for_model(model_name, ratio_for_api)
        size_for_api = (image_size or "2K").strip().upper()
        if size_for_api not in {"1K", "2K", "4K"}:
            size_for_api = "2K"

        seed_for_api = -1

        input_images_b64: List[str] = []
        ref_count = 0
        for i in range(1, 15):
            key_cn = f"参考图{i}"
            key_en = f"image_{i}"
            image_input = kwargs.get(key_cn, kwargs.get(key_en))
            if not isinstance(image_input, torch.Tensor):
                continue
            try:
                if image_input.dim() == 4:
                    for idx in range(image_input.shape[0]):
                        input_images_b64.append(_tensor_to_base64(image_input[idx : idx + 1]))
                        ref_count += 1
                else:
                    input_images_b64.append(_tensor_to_base64(image_input))
                    ref_count += 1
            except Exception as exc:
                _log(f"Gemini image encode failed: {key_cn} - {exc}")

        request_data = _create_reference_gemini_request(
            prompt=prompt,
            seed=seed_for_api,
            aspect_ratio=ratio_for_api,
            image_size=size_for_api,
            input_images_b64=input_images_b64,
            include_image_size=include_image_size,
            add_prompt_config_suffix=False,
            include_top_p=False,
        )

        _log(
            f"Gemini request: model={model_name}, size={size_for_api}, "
            f"aspect={ratio_for_api}, seed={seed_for_api}, refs={ref_count}, "
            f"include_image_size={include_image_size}"
        )
        task_payload = {
            "prompt": prompt,
            "api_key": (api_key or "").strip(),
            "model_name": model_name,
            "size_for_api": size_for_api,
            "ratio_for_api": ratio_for_api,
            "seed_for_api": seed_for_api,
            "ref_count": ref_count,
            "include_image_size": include_image_size,
            "request_data": request_data,
        }
        tensor, result_format, error_message = _execute_gemini_image_task(task_payload)
        if tensor is None:
            _log(f"Gemini failed. last_error={error_message}")
            return (_blank_image(), "unknown")
        hw = _get_image_hw(tensor)
        if hw:
            _log(f"Gemini output size: {hw[1]}x{hw[0]} (w x h)")
        return (tensor, result_format)


class KRGeminiImageAsyncSubmitNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Keep exactly the same params as sync Gemini node.
        return KRGeminiImageNode.INPUT_TYPES()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("任务信息",)
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME
    OUTPUT_NODE = True

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        _ensure_gemini_async_workers_started()
        task_payload, error = _prepare_gemini_image_task_from_kwargs(kwargs)
        if task_payload is None:
            return (f"[Comfyui-Kr-API] Gemini异步提交失败: {error}",)

        # 关键：在主线程做 POST 提交，避免子线程触发系统代理引发的 ProxyError。
        # 主线程 requests 已被 ComfyUI 预热，行为和同步节点完全一致。
        upstream_task_id, query_url, direct_tensor, submit_error = _submit_gemini_task_to_gateway(task_payload)

        task_id = str(uuid.uuid4())
        task_entry: Dict[str, Any] = {
            "task_id": task_id,
            "task_payload": task_payload,
            "upstream_task_id": upstream_task_id,
            "query_url": query_url,
            "status": "SUBMITTING",
            "result_tensor": direct_tensor if isinstance(direct_tensor, torch.Tensor) else None,
            "result_format": "unknown",
            "error": "",
            "created_at": time.time(),
            "bypass_proxy": bool(kwargs.get("绕过代理", True)),
        }

        # 提交本身就失败（401/网络错误等）：直接置为 FAILED，无需进队列
        if submit_error and not upstream_task_id and direct_tensor is None:
            task_entry["status"] = "FAILED"
            task_entry["error"] = submit_error
            with GEMINI_ASYNC_QUEUE_LOCK:
                GEMINI_ASYNC_TASK_QUEUE.append(task_entry)
                pending_count = len(GEMINI_ASYNC_TASK_QUEUE)
            _log(f"Gemini async submit failed: id={task_id}, error={submit_error}")
            return (
                f"Gemini异步提交失败\n"
                f"task_id: {task_id}\n"
                f"error: {submit_error}\n"
                f"queue: {pending_count}",
            )

        with GEMINI_ASYNC_QUEUE_LOCK:
            # 不论是否拿到 upstream_task_id（极少数直接返图情况也走这里）都丢给 worker 处理
            GEMINI_ASYNC_SUBMISSION_QUEUE.append(task_entry)
            GEMINI_ASYNC_TASK_QUEUE.append(task_entry)
            pending_count = len(GEMINI_ASYNC_TASK_QUEUE)

        _log(
            f"Gemini async submit: id={task_id}, upstream={upstream_task_id or 'direct-image'}, "
            f"model={task_payload['model_name']}, refs={task_payload['ref_count']}, queued={pending_count}"
        )
        message = (
            f"Gemini异步任务已提交\n"
            f"task_id: {task_id}\n"
            f"upstream_task_id: {upstream_task_id or 'N/A'}\n"
            f"model: {task_payload['model_name']}\n"
            f"refs: {task_payload['ref_count']}\n"
            f"queue: {pending_count}"
        )
        return (message,)


class KRGeminiImageAsyncFetchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "最多等待秒数": ("INT", {"default": 300, "min": 1, "max": 1800}),
                "绕过代理": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("图像", "结果格式", "响应信息")
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        max_wait_seconds = int(kwargs.get("最多等待秒数", 300))
        with GEMINI_ASYNC_QUEUE_LOCK:
            if not GEMINI_ASYNC_TASK_QUEUE:
                return (_blank_image(), "unknown", "Gemini异步队列为空")
            current_tasks = list(GEMINI_ASYNC_TASK_QUEUE)
            GEMINI_ASYNC_TASK_QUEUE.clear()

        images: List[torch.Tensor] = []
        formats: List[str] = []
        reports: List[str] = []
        requeue_tasks: List[Dict[str, Any]] = []
        wait_deadline = time.time() + float(max_wait_seconds)

        for task in current_tasks:
            task_id = str(task.get("task_id", ""))
            while True:
                status = str(task.get("status", ""))
                if status in {"DONE", "FAILED"}:
                    break
                if time.time() >= wait_deadline:
                    break
                time.sleep(0.2)

            status = str(task.get("status", ""))
            if status == "DONE":
                tensor = task.get("result_tensor")
                fmt = str(task.get("result_format", "unknown"))
                if isinstance(tensor, torch.Tensor):
                    images.append(tensor)
                    formats.append(fmt)
                    reports.append(f"{task_id}: DONE ({fmt})")
                else:
                    reports.append(f"{task_id}: DONE but empty tensor")
            elif status == "FAILED":
                reports.append(f"{task_id}: FAILED - {task.get('error', 'unknown error')}")
            else:
                # Still running; put back to queue for next fetch.
                requeue_tasks.append(task)
                reports.append(f"{task_id}: {status or 'RUNNING'} (requeued)")

        if requeue_tasks:
            with GEMINI_ASYNC_QUEUE_LOCK:
                GEMINI_ASYNC_TASK_QUEUE[0:0] = requeue_tasks

        if not images:
            return (_blank_image(), "unknown", "\n".join(reports) if reports else "无可用结果")

        summary_format = "mixed"
        if formats and all(fmt == formats[0] for fmt in formats):
            summary_format = formats[0]
        return (_stack_images(images), summary_format, "\n".join(reports))


def _validate_openai_size(size_value: str) -> Tuple[bool, str]:
    match = re.match(r"^(\d+)\s*x\s*(\d+)$", (size_value or "").strip())
    if not match:
        return False, "尺寸格式必须是 宽x高，例如 1536x864"

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return False, "尺寸宽高必须大于 0"
    if max(width, height) > 3840:
        return False, "尺寸长边不能超过 3840"
    if width % 16 != 0 or height % 16 != 0:
        return False, "尺寸宽高必须是 16 的倍数"
    if max(width, height) / float(min(width, height)) > 3.0:
        return False, "尺寸长宽比不能超过 3:1"
    pixels = width * height
    if pixels < 655_360 or pixels > 8_294_400:
        return False, "总像素必须在 655,360 到 8,294,400 之间"
    return True, ""


def _download_openai_url_bytes(url: str, api_key: str, timeout_sec: int = 120) -> Optional[bytes]:
    try:
        host = (urlparse(url).hostname or "").lower()
        # Pre-signed object storage URLs (e.g. s3) must be fetched without Authorization header.
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        }
        if host in {"ai.krapi.cn", "154.44.9.184", "localhost", "127.0.0.1"}:
            headers["Authorization"] = f"Bearer {(api_key or '').strip()}"
        response = _http_get(
            url,
            headers=headers,
            timeout=timeout_sec,
        )
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def _extract_openai_images_from_response(body: Any, api_key: str) -> Tuple[List[torch.Tensor], List[str]]:
    images: List[torch.Tensor] = []
    urls: List[str] = []
    seen: set[str] = set()

    for candidate in _collect_values(body):
        if candidate in seen:
            continue
        seen.add(candidate)

        if candidate.startswith(("http://", "https://")):
            raw = _download_openai_url_bytes(candidate, api_key)
            if raw:
                try:
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                    images.append(_pil_to_tensor(image))
                    urls.append(candidate)
                except Exception:
                    pass
            continue

        # Allow base64/b64_json responses as a fallback.
        raw = _decode_maybe_base64(candidate)
        if not raw:
            continue
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            images.append(_pil_to_tensor(image))
        except Exception:
            pass

    # Fallback: scan any text field for embedded URLs, e.g. markdown/image links.
    if not images:
        text_candidates: List[str] = []

        def _collect_texts(node: Any) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    _collect_texts(value)
            elif isinstance(node, list):
                for item in node:
                    _collect_texts(item)
            elif isinstance(node, str):
                text_candidates.append(node)

        _collect_texts(body)
        for text in text_candidates:
            url = _extract_url_from_text(text)
            if not url or url in seen:
                continue
            seen.add(url)
            raw = _download_openai_url_bytes(url, api_key)
            if not raw:
                continue
            try:
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                images.append(_pil_to_tensor(image))
                urls.append(url)
            except Exception:
                pass

    return images, urls


def _extract_openai_task_id(body: Any) -> Optional[str]:
    if not isinstance(body, dict):
        return None

    for key in ("task_id", "taskId"):
        value = body.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()

    data = body.get("data")
    if isinstance(data, dict):
        for key in ("task_id", "taskId"):
            value = data.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
    return None


def _extract_openai_status_and_payload(body: Any) -> Tuple[str, Any]:
    if not isinstance(body, dict):
        return "unknown", body

    status = str(body.get("status", "unknown"))
    payload: Any = body.get("data", body)

    if isinstance(payload, dict):
        if payload.get("status") is not None:
            status = str(payload.get("status"))
        if payload.get("data") is not None:
            payload = payload.get("data")

    return status.lower(), payload


def _poll_openai_task_result(api_key: str, task_id: str, max_attempts: int = 90, interval_sec: float = 2.0) -> Any:
    query_url = _image_task_query_url(task_id)
    last_body: Any = None
    for _ in range(max_attempts):
        response = _http_get(query_url, headers=_make_headers(api_key), timeout=(15, 120))
        if response.status_code != 200:
            time.sleep(interval_sec)
            continue

        try:
            body = response.json()
        except Exception:
            body = {"raw_text": (response.text or "")[:800]}
        last_body = body

        status, payload = _extract_openai_status_and_payload(body)
        if status in {"success", "completed", "done", "finished"}:
            return payload
        if status in {"failed", "error", "failure"}:
            raise RuntimeError(f"任务失败: {json.dumps(body, ensure_ascii=False)[:400]}")
        time.sleep(interval_sec)

    if last_body is not None:
        raise RuntimeError(f"任务轮询超时: {json.dumps(last_body, ensure_ascii=False)[:400]}")
    raise RuntimeError("任务轮询超时: 未获取到任何状态响应")


def _extract_video_url_from_payload(payload: Any) -> Optional[str]:
    def _normalize_video_url(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        text = value.strip().strip("\"'")
        if not text:
            return None
        text = text.replace("\\/", "/")
        text = re.sub(r"^https?://\s+", "https://", text, flags=re.IGNORECASE)
        text = re.sub(r"^https?://%20", "https://", text, flags=re.IGNORECASE)
        text = text.replace("https:// ai.krapi.cn", "https://ai.krapi.cn")
        text = text.replace("http:// ai.krapi.cn", "https://ai.krapi.cn")
        text = text.replace("https://%20ai.krapi.cn", "https://ai.krapi.cn")
        text = text.replace("http://%20ai.krapi.cn", "https://ai.krapi.cn")
        if text.startswith("/"):
            return VEO_API_BASE.rstrip("/") + text
        if text.startswith("ai.krapi.cn/"):
            return "https://" + text
        if text.startswith("http://ai.krapi.cn/"):
            return "https://" + text[len("http://"):]
        return text

    def _is_http_url(value: Any) -> bool:
        text = _normalize_video_url(value)
        if not text:
            return False
        return text.startswith(("http://", "https://"))

    def _is_video_url(value: Any) -> bool:
        text = _normalize_video_url(value)
        if not text:
            return False
        text = text.lower()
        return text.startswith(("http://", "https://")) and any(
            ext in text for ext in (".mp4", ".mov", ".webm", ".m3u8")
        )

    def _extract_urls_from_text(value: str) -> List[str]:
        text = (value or "").strip()
        if not text:
            return []
        text_norm = text.replace("\\/", "/")
        found = re.findall(r"https?://[^\s\"'<>\\]+", text_norm)
        return [u.strip() for u in found if u.strip()]

    if isinstance(payload, dict):
        for key in ("video_url", "videoUrl", "output", "video", "url", "download_url"):
            value = payload.get(key)
            if _is_http_url(value):
                return _normalize_video_url(value)

        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("video_url", "videoUrl", "output", "video", "url", "download_url"):
                value = data.get(key)
                if _is_http_url(value):
                    return _normalize_video_url(value)
            nested_output = data.get("output")
            if isinstance(nested_output, dict):
                for key in ("video_url", "url"):
                    value = nested_output.get(key)
                    if _is_http_url(value):
                        return _normalize_video_url(value)

    candidates: List[str] = []
    url_like_candidates: List[str] = []

    def _walk(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_l = str(key).strip().lower()
                if isinstance(value, str):
                    text = _normalize_video_url(value) or ""
                    if _is_video_url(text):
                        candidates.append(text)
                    elif text.startswith(("http://", "https://")) and any(
                        token in key_l for token in ("url", "video", "content", "file", "download", "result", "media")
                    ):
                        url_like_candidates.append(text)
                    else:
                        for u in _extract_urls_from_text(text):
                            if _is_video_url(u):
                                candidates.append(u)
                            elif any(
                                token in key_l for token in ("url", "video", "content", "file", "download", "result", "media")
                            ):
                                url_like_candidates.append(u)
                _walk(value, key_l)
        elif isinstance(node, list):
            for item in node:
                _walk(item, parent_key)
        elif isinstance(node, str):
            text = _normalize_video_url(node) or ""
            if _is_video_url(text):
                candidates.append(text)
            elif text.startswith(("http://", "https://")) and any(
                token in parent_key for token in ("url", "video", "content", "file", "download", "result", "media")
            ):
                url_like_candidates.append(text)
            else:
                for u in _extract_urls_from_text(text):
                    if _is_video_url(u):
                        candidates.append(u)
                    elif any(
                        token in parent_key for token in ("url", "video", "content", "file", "download", "result", "media")
                    ):
                        url_like_candidates.append(u)

    _walk(payload)
    if candidates:
        return candidates[0]
    if url_like_candidates:
        for u in url_like_candidates:
            ul = u.lower()
            if "/v1/videos/" in ul and (ul.endswith("/content") or "/content?" in ul):
                continue
            if "/v1/videos/" in ul and ("task" in ul or "/v1/videos/" in ul):
                continue
            return u
        return url_like_candidates[0]
    return None


def _extract_veo_task_id(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ("task_id", "taskId", "id"):
        value = payload.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("task_id", "taskId", "id"):
            value = data.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value).strip()
    return None


def _extract_veo_status(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    for key in ("status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("status", "state"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()

    return ""


def _poll_veo_video_task(api_key: str, task_id: str, max_attempts: int = 300, interval_sec: float = 2.0) -> Dict[str, Any]:
    last_payload: Dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        response = _http_get(f"{VEO_VIDEO_QUERY_URL}/{task_id}", headers=_make_headers(api_key), timeout=(15, 120))

        if response.status_code != 200:
            _log(
                f"Veo poll failed: attempt={attempt}, http={response.status_code}, "
                f"body={(response.text or '')[:220]}"
            )
            time.sleep(interval_sec)
            continue
        try:
            payload = response.json()
        except Exception:
            payload = {"raw_text": (response.text or "")[:800]}
        last_payload = payload if isinstance(payload, dict) else {"raw": str(payload)[:800]}

        status = _extract_veo_status(last_payload)
        video_url = _extract_video_url_from_payload(last_payload)
        _log(f"Veo poll: attempt={attempt}, status={status or 'UNKNOWN'}, has_video_url={'yes' if video_url else 'no'}")

        if status in {"SUCCESS", "COMPLETED", "DONE", "FINISHED", "SUCCEEDED"}:
            return last_payload
        if video_url and status not in {"SUBMITTED", "PROCESSING", "PENDING", "QUEUED", "IN_PROGRESS", "RUNNING", "CREATED"}:
            # 有 video_url 且 status 不是明确的"进行中"状态,认为完成
            return last_payload
        if status in {"FAILURE", "FAILED", "ERROR", "CANCELLED", "REJECTED"}:
            raise RuntimeError(f"Veo任务失败: {json.dumps(last_payload, ensure_ascii=False)[:500]}")
        time.sleep(interval_sec)

    raise RuntimeError(f"Veo任务轮询超时: {json.dumps(last_payload, ensure_ascii=False)[:500]}")


def _poll_grok_video_task(api_key: str, task_id: str, max_attempts: int = 300, interval_sec: float = 2.0) -> Dict[str, Any]:
    last_payload: Dict[str, Any] = {}
    query_url = f"{GROK_VIDEO_QUERY_URL}/{task_id}"
    req_headers = {"Authorization": f"Bearer {(api_key or '').strip()}"}
    for attempt in range(1, max_attempts + 1):
        response = _http_get(query_url, headers=req_headers, timeout=(15, 120))
        if response.status_code != 200:
            failed_code = response.status_code
            failed_body = (response.text or "")[:220]
            _log(f"Grok poll failed: attempt={attempt}, http={failed_code}, body={failed_body}")
            time.sleep(interval_sec)
            continue

        try:
            payload = response.json()
        except Exception:
            payload = {"raw_text": (response.text or "")[:800]}
        last_payload = payload if isinstance(payload, dict) else {"raw": str(payload)[:800]}

        status = _extract_veo_status(last_payload)
        video_url = _extract_video_url_from_payload(last_payload)
        _log(f"Grok poll: attempt={attempt}, status={status or 'UNKNOWN'}, has_video_url={'yes' if video_url else 'no'}")

        if status in {"SUCCESS", "COMPLETED", "DONE", "FINISHED", "SUCCEEDED"} or video_url:
            return last_payload
        if status in {"FAILURE", "FAILED", "ERROR", "CANCELLED", "REJECTED"}:
            raise RuntimeError(f"Grok任务失败: {json.dumps(last_payload, ensure_ascii=False)[:500]}")
        time.sleep(interval_sec)

    raise RuntimeError(f"Grok任务轮询超时: {json.dumps(last_payload, ensure_ascii=False)[:500]}")


def _extract_kling_status(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    data = payload.get("data")
    if isinstance(data, dict):
        status = data.get("task_status")
        if isinstance(status, str) and status.strip():
            return status.strip().lower()
    for key in ("task_status", "status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _extract_kling_video_url(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return _extract_video_url_from_payload(payload)
    data = payload.get("data")
    if isinstance(data, dict):
        task_result = data.get("task_result")
        if isinstance(task_result, dict):
            videos = task_result.get("videos")
            if isinstance(videos, list):
                for item in videos:
                    if isinstance(item, dict):
                        url = item.get("url")
                        if isinstance(url, str) and url.strip().startswith(("http://", "https://")):
                            return url.strip()
        url = data.get("video_url") or data.get("url")
        if isinstance(url, str) and url.strip().startswith(("http://", "https://")):
            return url.strip()
    return _extract_video_url_from_payload(payload)


def _poll_kling_image2video_task(api_key: str, task_id: str, max_attempts: int = 300, interval_sec: float = 2.0) -> Dict[str, Any]:
    last_payload: Dict[str, Any] = {}
    query_url = f"{KLING_IMAGE2VIDEO_QUERY_URL}/{task_id}"
    for attempt in range(1, max_attempts + 1):
        response = _http_get(query_url, headers=_make_headers(api_key), timeout=(15, 120))
        if response.status_code != 200:
            _log(f"Kling poll failed: attempt={attempt}, http={response.status_code}, body={(response.text or '')[:220]}")
            time.sleep(interval_sec)
            continue
        try:
            payload = response.json()
        except Exception:
            payload = {"raw_text": (response.text or "")[:800]}
        last_payload = payload if isinstance(payload, dict) else {"raw": str(payload)[:800]}

        status = _extract_kling_status(last_payload)
        video_url = _extract_kling_video_url(last_payload)
        _log(f"Kling poll: attempt={attempt}, status={status or 'unknown'}, has_video_url={'yes' if video_url else 'no'}")

        if status in {"succeed", "success", "completed", "done", "finished"} or video_url:
            return last_payload
        if status in {"failed", "failure", "error", "cancelled", "rejected"}:
            raise RuntimeError(f"可灵任务失败: {json.dumps(last_payload, ensure_ascii=False)[:500]}")
        time.sleep(interval_sec)

    raise RuntimeError(f"可灵任务轮询超时: {json.dumps(last_payload, ensure_ascii=False)[:500]}")



def _openai_image_async_worker_loop(worker_name: str) -> None:
    while True:
        task: Optional[Dict[str, Any]] = None
        with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
            if OPENAI_IMAGE_ASYNC_SUBMISSION_QUEUE:
                task = OPENAI_IMAGE_ASYNC_SUBMISSION_QUEUE.pop(0)
        if task is None:
            time.sleep(0.1)
            continue

        try:
            with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
                task["status"] = "RUNNING"
                task["started_at"] = time.time()

            api_key = str(task.get("api_key", "") or "").strip()
            upstream_task_id = str(task.get("upstream_task_id", "") or "")
            query_url = str(task.get("query_url", "") or "")

            _log(f"OpenAI async[{worker_name}] start polling: id={task.get('task_id')}, upstream={upstream_task_id}")

            if not upstream_task_id and not query_url:
                pre_tensor = task.get("result_tensor")
                if isinstance(pre_tensor, torch.Tensor):
                    with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
                        task["status"] = "DONE"
                        task["result_format"] = "url"
                        task["finished_at"] = time.time()
                    _log(f"OpenAI async[{worker_name}] done(direct): id={task.get('task_id')}")
                    continue
                with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
                    task["status"] = "FAILED"
                    task["error"] = "missing upstream task_id"
                    task["finished_at"] = time.time()
                continue

            tensor, err = _finalize_gemini_task_from_query(api_key, upstream_task_id, query_url)
            with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
                if tensor is not None:
                    task["status"] = "DONE"
                    task["result_tensor"] = tensor
                    task["result_format"] = "url"
                    task["error"] = ""
                    task["finished_at"] = time.time()
                    _log(f"OpenAI async[{worker_name}] done: id={task.get('task_id')}")
                else:
                    task["status"] = "FAILED"
                    task["error"] = err
                    task["finished_at"] = time.time()
                    _log(f"OpenAI async[{worker_name}] failed: id={task.get('task_id')}, error={err}")
        except Exception as exc:
            with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
                task["status"] = "FAILED"
                task["error"] = f"worker exception: {exc}"
                task["finished_at"] = time.time()
            _log(f"OpenAI async[{worker_name}] exception: {exc}")


def _ensure_openai_image_async_workers_started() -> None:
    global OPENAI_IMAGE_ASYNC_WORKERS_STARTED
    if OPENAI_IMAGE_ASYNC_WORKERS_STARTED:
        return
    with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
        if OPENAI_IMAGE_ASYNC_WORKERS_STARTED:
            return
        for i in range(OPENAI_IMAGE_ASYNC_WORKER_COUNT):
            thread = threading.Thread(
                target=_openai_image_async_worker_loop,
                args=(f"OW{i + 1}",),
                daemon=True,
            )
            thread.start()
        OPENAI_IMAGE_ASYNC_WORKERS_STARTED = True
        _log(f"OpenAI image async workers started: {OPENAI_IMAGE_ASYNC_WORKER_COUNT}")


class KRGPTImage2AsyncSubmitNode:
    @classmethod
    def INPUT_TYPES(cls):
        return KRGPTImage2Node.INPUT_TYPES()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("任务信息",)
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME
    OUTPUT_NODE = True

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        _ensure_openai_image_async_workers_started()

        prompt = (kwargs.get("提示词", "") or "").strip()
        aspect_ratio = (kwargs.get("比例", "自动") or "").strip()
        resolution = (kwargs.get("分辨率", "2K") or "").strip()
        quality = (kwargs.get("质量", "高") or "").strip()
        quality_map = {
            "标准": "standard",
            "hd": "high",
            "高": "high",
            "中等": "medium",
            "低": "low",
        }
        quality_value = quality_map.get(quality, "high")
        output_format = (kwargs.get("输出格式", "png") or "").strip()
        seed = int(kwargs.get("种子", 0))
        api_key = (kwargs.get("API密钥", "") or "").strip()

        if not api_key:
            return ("API密钥为空，请填写后再试",)
        if not prompt:
            return ("提示词为空",)

        # 收集参考图
        refs: List[torch.Tensor] = []
        for i in range(1, 15):
            img = kwargs.get(f"参考图{i}")
            if isinstance(img, torch.Tensor):
                refs.append(img)

        # 比例为"自动"时,如果有参考图,根据第一张参考图的宽高比自动匹配
        if aspect_ratio in {"自动", "auto", ""}:
            if refs:
                best = _choose_best_aspect_ratio_from_options(refs[0], GPT_IMAGE2_ASPECT_RATIO_OPTIONS)
                if best and best not in {"自动"}:
                    aspect_ratio = best
                    _log(f"GPT-Image-2 async auto ratio from reference image: {aspect_ratio}")

        size = _gpt_image2_resolve_size(aspect_ratio, resolution)

        # 构造 payload
        form_data: Optional[Dict[str, str]] = None
        files: Optional[List[Tuple[str, Tuple[str, bytes, str]]]] = None
        if refs:
            form_data, files = _build_gpt_image2_edit_multipart(
                prompt,
                size,
                aspect_ratio,
                resolution,
                quality_value,
                output_format,
                refs,
            )
            payload = None
            path = "/v1/images/edits"
        else:
            payload = {
                "model": "gpt-image-2",
                "prompt": prompt,
                "size": size,
                "ratio": aspect_ratio,
                "image_size": resolution,
                "n": 1,
                "quality": quality_value,
                "output_format": output_format,
            }
            path = "/v1/images/generations"

        submit_url = OPENAI_API_ROOT.rstrip("/") + path

        _log(f"GPT-Image-2 async submit: path={path}, size={size}, refs={len(refs)}")

        try:
            if files is not None:
                resp = _http_post(
                    submit_url,
                    headers=_make_multipart_headers(api_key),
                    data=form_data or {},
                    files=files,
                    timeout=(120, 600),
                )
            else:
                resp = _http_post(submit_url, headers=_make_headers(api_key), json=payload, timeout=(120, 600))
        except Exception as exc:
            return (f"提交失败: {exc}",)

        if resp.status_code not in (200, 201, 202):
            return (f"提交失败: HTTP {resp.status_code}: {(resp.text or '')[:300]}",)

        try:
            submit_body = resp.json()
        except Exception:
            return (f"提交失败: non-json response",)

        # 提取 task_id
        upstream_task_id, query_url = _extract_native_image_task_info(submit_body)
        if not upstream_task_id and not query_url:
            upstream_task_id, query_url = _extract_async_task_info_from_chat_response(submit_body)

        local_task_id = str(uuid.uuid4())
        task_entry: Dict[str, Any] = {
            "task_id": local_task_id,
            "api_key": api_key,
            "upstream_task_id": upstream_task_id,
            "query_url": query_url,
            "status": "SUBMITTING",
            "result_tensor": None,
            "result_format": "unknown",
            "error": "",
            "created_at": time.time(),
            "bypass_proxy": bool(kwargs.get("绕过代理", True)),
        }

        # 可能直接返回了图片
        if not upstream_task_id and not query_url:
            if isinstance(submit_body, dict) and submit_body.get("data"):
                data_list = submit_body.get("data", [])
                if isinstance(data_list, list) and data_list:
                    item = data_list[0]
                    b64 = item.get("b64_json")
                    if b64:
                        tensor = _decode_base64_image_to_tensor(b64)
                        if tensor is not None:
                            task_entry["result_tensor"] = tensor
                            task_entry["status"] = "DONE"
                            task_entry["result_format"] = "url"

            direct_images, _ = _extract_openai_images_from_response(submit_body, api_key)
            if direct_images and task_entry["status"] != "DONE":
                task_entry["result_tensor"] = direct_images[0]
                task_entry["status"] = "DONE"
                task_entry["result_format"] = "url"

        with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
            OPENAI_IMAGE_ASYNC_SUBMISSION_QUEUE.append(task_entry)
            OPENAI_IMAGE_ASYNC_TASK_QUEUE.append(task_entry)
            pending_count = len(OPENAI_IMAGE_ASYNC_TASK_QUEUE)

        _log(f"GPT-Image-2 async task queued: id={local_task_id}, upstream={upstream_task_id or 'direct'}")

        message = (
            f"GPT-Image-2异步任务已提交\n"
            f"task_id: {local_task_id}\n"
            f"size: {size}\n"
            f"refs: {len(refs)}\n"
            f"queue: {pending_count}"
        )
        return (message,)


class KRGPTImage2AsyncFetchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "最多等待秒数": ("INT", {"default": 300, "min": 1, "max": 1800}),
                "绕过代理": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "响应信息")
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        max_wait_seconds = int(kwargs.get("最多等待秒数", 300))
        with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
            if not OPENAI_IMAGE_ASYNC_TASK_QUEUE:
                return (_blank_image(), "GPT-Image-2异步队列为空")
            current_tasks = list(OPENAI_IMAGE_ASYNC_TASK_QUEUE)
            OPENAI_IMAGE_ASYNC_TASK_QUEUE.clear()

        images: List[torch.Tensor] = []
        reports: List[str] = []
        requeue_tasks: List[Dict[str, Any]] = []
        wait_deadline = time.time() + float(max_wait_seconds)

        for task in current_tasks:
            task_id = str(task.get("task_id", ""))
            while True:
                status = str(task.get("status", ""))
                if status in {"DONE", "FAILED"}:
                    break
                if time.time() >= wait_deadline:
                    break
                time.sleep(0.2)

            status = str(task.get("status", ""))
            if status == "DONE":
                tensor = task.get("result_tensor")
                if isinstance(tensor, torch.Tensor):
                    images.append(tensor)
                    reports.append(f"{task_id}: DONE")
                else:
                    reports.append(f"{task_id}: DONE but empty tensor")
            elif status == "FAILED":
                reports.append(f"{task_id}: FAILED - {task.get('error', 'unknown')}")
            else:
                requeue_tasks.append(task)
                reports.append(f"{task_id}: {status or 'RUNNING'} (requeued)")

        if requeue_tasks:
            with OPENAI_IMAGE_ASYNC_QUEUE_LOCK:
                OPENAI_IMAGE_ASYNC_TASK_QUEUE[0:0] = requeue_tasks

        if not images:
            return (_blank_image(), "\n".join(reports) if reports else "无可用结果")

        return (_stack_images(images), "\n".join(reports))


VEO_API_BASE = "https://ai.krapi.cn"


def _veo_submit_and_poll(api_key: str, model: str, payload: Dict[str, Any], max_poll: int = 300, interval: float = 15.0) -> Tuple[Optional[str], str]:
    submit_url = f"{VEO_API_BASE.rstrip('/')}/v1/videos"
    headers = _make_headers(api_key)

    _log(
        f"Veo submit: url={submit_url}, model={model}, "
        f"images={_count_veo_message_images(payload)}, seconds={payload.get('seconds')}"
    )

    try:
        resp = _http_post(submit_url, headers=headers, json=payload, timeout=(120, 600))
    except Exception as exc:
        return None, f"submit exception: {exc}"

    _log(f"Veo submit result: http={resp.status_code}, body={resp.text[:500]}")

    if resp.status_code not in (200, 201, 202):
        return None, f"submit HTTP {resp.status_code}: {(resp.text or '')[:400]}"

    try:
        submit_body = resp.json()
    except Exception:
        return None, f"submit non-json: {(resp.text or '')[:300]}"

    task_id = _extract_veo_task_id(submit_body) or ""

    if not task_id:
        video_url = _extract_video_url_from_payload(submit_body)
        if video_url:
            return video_url, ""
        return None, f"no task_id in submit response: {json.dumps(submit_body, ensure_ascii=False)[:400]}"

    _log(f"Veo task created: {task_id}, polling (interval={interval}s)...")

    query_url = f"{VEO_API_BASE.rstrip('/')}/v1/videos/{task_id}"
    last_payload: Dict[str, Any] = {}

    for attempt in range(1, max_poll + 1):
        time.sleep(interval)
        try:
            poll_resp = _http_get(
                query_url,
                headers={"Authorization": f"Bearer {(api_key or '').strip()}"},
                params={"model": model},
                timeout=(15, 120),
            )
        except Exception as exc:
            _log(f"Veo poll exception: attempt={attempt}, error={exc}")
            continue

        if poll_resp.status_code != 200:
            _log(f"Veo poll failed: attempt={attempt}, http={poll_resp.status_code}, body={poll_resp.text[:220]}")
            continue

        try:
            poll_body = poll_resp.json()
        except Exception:
            poll_body = {"raw": (poll_resp.text or "")[:500]}

        last_payload = poll_body if isinstance(poll_body, dict) else {}

        status = str(last_payload.get("status", "unknown")).strip().lower()
        video_url = _extract_video_url_from_payload(last_payload)

        _log(f"Veo poll: attempt={attempt}, status={status}, has_video_url={'yes' if video_url else 'no'}")

        if status in {"completed", "complete", "done", "success", "succeeded"}:
            if video_url:
                return video_url, ""
            return f"{query_url}/content", ""

        if status in {"failed", "failure", "error", "cancelled", "canceled", "rejected"}:
            error = last_payload.get("error", {})
            err_msg = error.get("message", "") if isinstance(error, dict) else str(error)
            return None, f"task failed: {err_msg or json.dumps(last_payload, ensure_ascii=False)[:400]}"

    return None, f"poll timeout after {max_poll} attempts: {json.dumps(last_payload, ensure_ascii=False)[:400]}"


def _veo_download_video(video_url: str, api_key: str) -> Optional[str]:
    """下载视频到临时文件,返回文件路径。"""
    try:
        tmp_dir = os.path.join(tempfile.gettempdir(), "comfyui_kr_api_videos")
        os.makedirs(tmp_dir, exist_ok=True)
        file_path = os.path.join(tmp_dir, f"veo_{int(time.time() * 1000)}.mp4")
        clean_url = (video_url or "").strip().strip("\"'")
        clean_url = clean_url.replace("\\/", "/")
        clean_url = re.sub(r"^https?://\s+", "https://", clean_url, flags=re.IGNORECASE)
        clean_url = re.sub(r"^https?://%20", "https://", clean_url, flags=re.IGNORECASE)
        clean_url = clean_url.replace("https:// ai.krapi.cn", "https://ai.krapi.cn")
        clean_url = clean_url.replace("http:// ai.krapi.cn", "https://ai.krapi.cn")
        clean_url = clean_url.replace("https://%20ai.krapi.cn", "https://ai.krapi.cn")
        clean_url = clean_url.replace("http://%20ai.krapi.cn", "https://ai.krapi.cn")
        if clean_url.startswith("/"):
            clean_url = VEO_API_BASE.rstrip("/") + clean_url
        elif clean_url.startswith("ai.krapi.cn/"):
            clean_url = "https://" + clean_url
        elif clean_url.startswith("http://ai.krapi.cn/"):
            clean_url = "https://" + clean_url[len("http://"):]
        url_l = clean_url.lower()
        headers = {"Authorization": f"Bearer {(api_key or '').strip()}"} if "/v1/videos/" in url_l else {}
        resp = _http_get(clean_url, headers=headers, stream=True, timeout=(30, 600))
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        _log(f"Veo video downloaded: {file_path}")
        return file_path
    except Exception as exc:
        _log(f"Veo video download failed: {exc}")
        return None


def _count_veo_message_images(payload: Dict[str, Any]) -> int:
    count = 0
    for message in payload.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                count += 1
    return count


def _build_veo_messages(image_urls: List[str], prompt: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": url}}
        for url in image_urls
        if isinstance(url, str) and url.strip()
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class KRVeoImageToVideoNode:
    """Veo 图生视频节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "模型": ("STRING", {"multiline": False, "default": "veo_3_1_i2v_fast_landscape"}),
                "时长": (VEO_DURATION_OPTIONS, {"default": "6秒"}),
                "参考图": ("IMAGE",),
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "最大轮询次数": ("INT", {"default": 300, "min": 1, "max": 3000}),
                "轮询间隔秒": ("INT", {"default": 15, "min": 1, "max": 60}),
                "绕过代理": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("视频", "信息")
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        prompt = (kwargs.get("提示词", "") or "").strip()
        model = (kwargs.get("模型", "veo_3_1_i2v_fast_landscape") or "").strip()
        duration_seconds = _parse_veo_duration_seconds(str(kwargs.get("时长", "6秒")))
        reference_image = kwargs.get("参考图")
        api_key = (kwargs.get("API密钥", "") or "").strip()
        max_poll = int(kwargs.get("最大轮询次数", 300))
        interval = int(kwargs.get("轮询间隔秒", 15))

        if not api_key:
            return (KRVideoAdapter(""), "API密钥为空")

        if not isinstance(reference_image, torch.Tensor):
            return (KRVideoAdapter(""), "Veo图生视频需要提供参考图")

        prompt_text = prompt or "生成视频"
        image_url = _tensor_to_compressed_jpeg_data_url(reference_image, max_long_side=1920, quality=100)
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt_text,
            "seconds": str(duration_seconds),
            "messages": _build_veo_messages([image_url], prompt_text),
        }

        _log(f"Veo i2v request: model={model}, refs=1, seconds={duration_seconds}")

        video_url, error = _veo_submit_and_poll(api_key, model, payload, max_poll=max_poll, interval=float(interval))

        if not video_url:
            _log(f"Veo failed: {error}")
            return (KRVideoAdapter(""), f"失败: {error}")

        # 下载视频
        file_path = _veo_download_video(video_url, api_key)
        if not file_path:
            return (KRVideoAdapter(""), f"视频下载失败,URL: {video_url}")

        if VideoFromFile is not None:
            video_output = VideoFromFile(file_path)
        else:
            video_output = KRVideoAdapter(file_path)

        return (video_output, f"成功: {video_url}")


class KRVeoKeyframeVideoNode:
    """Veo 首尾帧视频节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "模型": ("STRING", {"multiline": False, "default": "veo_3_1_i2v_fast_landscape"}),
                "时长": (VEO_DURATION_OPTIONS, {"default": "6秒"}),
                "首帧图": ("IMAGE",),
                "尾帧图": ("IMAGE",),
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "最大轮询次数": ("INT", {"default": 300, "min": 1, "max": 3000}),
                "轮询间隔秒": ("INT", {"default": 15, "min": 1, "max": 60}),
                "绕过代理": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (IO.VIDEO, "STRING")
    RETURN_NAMES = ("视频", "信息")
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        prompt = (kwargs.get("提示词", "") or "").strip()
        model = (kwargs.get("模型", "veo_3_1_i2v_fast_landscape") or "").strip()
        duration_seconds = _parse_veo_duration_seconds(str(kwargs.get("时长", "6秒")))
        api_key = (kwargs.get("API密钥", "") or "").strip()
        max_poll = int(kwargs.get("最大轮询次数", 300))
        interval = int(kwargs.get("轮询间隔秒", 15))
        first_frame = kwargs.get("首帧图")
        last_frame = kwargs.get("尾帧图")

        if not api_key:
            return (KRVideoAdapter(""), "API密钥为空")

        if not isinstance(first_frame, torch.Tensor):
            return (KRVideoAdapter(""), "首尾帧模式至少需要提供首帧图")
        if not isinstance(last_frame, torch.Tensor):
            return (KRVideoAdapter(""), "首尾帧模式需要提供尾帧图")

        prompt_text = prompt or "动起来"
        first_url = _tensor_to_compressed_jpeg_data_url(first_frame, max_long_side=1920, quality=100)
        last_url = _tensor_to_compressed_jpeg_data_url(last_frame, max_long_side=1920, quality=100)
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt_text,
            "seconds": str(duration_seconds),
            "messages": _build_veo_messages([first_url, last_url], prompt_text),
        }

        _log(f"Veo keyframe request: model={model}, refs=2, seconds={duration_seconds}")

        video_url, error = _veo_submit_and_poll(api_key, model, payload, max_poll=max_poll, interval=float(interval))

        if not video_url:
            _log(f"Veo keyframe failed: {error}")
            return (KRVideoAdapter(""), f"失败: {error}")

        file_path = _veo_download_video(video_url, api_key)
        if not file_path:
            return (KRVideoAdapter(""), f"视频下载失败,URL: {video_url}")

        if VideoFromFile is not None:
            video_output = VideoFromFile(file_path)
        else:
            video_output = KRVideoAdapter(file_path)

        return (video_output, f"成功: {video_url}")


# ================================
# GPT-Image-2 节点
# ================================

GPT_IMAGE2_SIZE_OPTIONS = [
    "1024x1024",
    "2048x2048",
    "2048x1152",
    "2048x1536",
    "1536x2048",
    "3840x2160",
    "2160x3840",
]

GPT_IMAGE2_ASPECT_RATIO_OPTIONS = [
    "自动",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]

GPT_IMAGE2_RESOLUTION_OPTIONS = [
    "1K",
    "2K",
    "4K",
]

GPT_IMAGE2_QUALITY_OPTIONS = [
    "标准",
    "高",
    "hd",
    "高",
    "中等",
    "低",
]

GPT_IMAGE2_FORMAT_OPTIONS = [
    "png",
    "jpeg",
    "webp",
]


def _gpt_image2_resolve_size(aspect_ratio: str, resolution: str) -> str:
    """根据比例 + 分辨率档位,计算最优尺寸。
    规则:
      - 宽高都是 16 的倍数
      - 长边不超过 3840
      - 总像素在 655,360 ~ 8,294,400 之间
      - 1K/2K/4K 分别对应约 1M/4M/8M 像素
    """
    import math

    # 目标总像素
    pixel_targets = {"1K": 1_048_576, "2K": 4_194_304, "4K": 8_294_400}
    max_pixels = 8_294_400
    min_pixels = 655_360
    max_long_side = 3840

    ratio = (aspect_ratio or "").strip()
    if ratio in {"自动", "auto", ""}:
        ratio = "1:1"

    res = (resolution or "2K").strip().upper()
    if res not in {"1K", "2K", "4K"}:
        res = "2K"

    # 解析比例
    parts = ratio.split(":")
    if len(parts) != 2:
        return "2048x2048"
    try:
        w_ratio = float(parts[0])
        h_ratio = float(parts[1])
    except ValueError:
        return "2048x2048"
    if w_ratio <= 0 or h_ratio <= 0:
        return "2048x2048"

    target_pixels = pixel_targets[res]

    # 计算: w = sqrt(P * w_ratio / h_ratio), h = sqrt(P * h_ratio / w_ratio)
    w = math.sqrt(target_pixels * w_ratio / h_ratio)
    h = math.sqrt(target_pixels * h_ratio / w_ratio)

    # 取 16 的倍数
    w = max(16, round(w / 16) * 16)
    h = max(16, round(h / 16) * 16)

    # 长边不超过 3840
    long_side = max(w, h)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        w = max(16, round(w * scale / 16) * 16)
        h = max(16, round(h * scale / 16) * 16)

    # 总像素检查
    while w * h > max_pixels:
        if w >= h:
            w -= 16
        else:
            h -= 16
        w = max(16, w)
        h = max(16, h)

    while w * h < min_pixels:
        if w <= h:
            w += 16
        else:
            h += 16

    return f"{int(w)}x{int(h)}"


def _build_gpt_image2_edit_multipart(
    prompt: str,
    size: str,
    aspect_ratio: str,
    resolution: str,
    quality: str,
    output_format: str,
    refs: List[torch.Tensor],
) -> Tuple[Dict[str, str], List[Tuple[str, Tuple[str, bytes, str]]]]:
    form_data: Dict[str, str] = {
        "model": "gpt-image-2",
        "prompt": prompt,
        "n": "1",
        "size": size,
        "quality": quality,
        "response_format": "url",
    }
    if output_format:
        form_data["output_format"] = output_format

    files: List[Tuple[str, Tuple[str, bytes, str]]] = []
    for index, ref in enumerate(refs):
        field = "image" if len(refs) == 1 else "image[]"
        filename = f"image-{index + 1}.jpg"
        files.append(
            (
                field,
                (
                    filename,
                    _tensor_to_compressed_jpeg_bytes(ref, max_long_side=2048, quality=100),
                    "image/jpeg",
                ),
            )
        )

    return form_data, files


class KRGPTImage2Node:
    """GPT-Image-2 生图节点。
    走 /v1/images/generations(文生图）或 /v1/images/edits（图生图）。
    原生异步：提交 → task_id → 轮询 /v1/images/tasks/{id} 拿结果。
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional_images = {f"参考图{i}": ("IMAGE",) for i in range(1, 15)}
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "比例": (GPT_IMAGE2_ASPECT_RATIO_OPTIONS, {"default": "自动"}),
                "分辨率": (GPT_IMAGE2_RESOLUTION_OPTIONS, {"default": "2K"}),
                "质量": (GPT_IMAGE2_QUALITY_OPTIONS, {"default": "高"}),
                "输出格式": (GPT_IMAGE2_FORMAT_OPTIONS, {"default": "png"}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "最大等待秒数": ("INT", {"default": 600, "min": 30, "max": 3600}),
                "API密钥": ("STRING", {"multiline": False, "default": ""}),
                "绕过代理": ("BOOLEAN", {"default": True}),
            },
            "optional": optional_images,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME

    def run(self, **kwargs):
        _set_bypass_proxy(bool(kwargs.get("绕过代理", True)))
        prompt = (kwargs.get("提示词", "") or "").strip()
        aspect_ratio = (kwargs.get("比例", "自动") or "").strip()
        resolution = (kwargs.get("分辨率", "2K") or "").strip()
        quality = (kwargs.get("质量", "高") or "").strip()
        quality_map = {
            "标准": "standard",
            "hd": "high",
            "高": "high",
            "中等": "medium",
            "低": "low",
        }
        quality_value = quality_map.get(quality, "high")
        output_format = (kwargs.get("输出格式", "png") or "").strip()
        seed = int(kwargs.get("种子", 0))
        max_wait = int(kwargs.get("最大等待秒数", 600))
        api_key = (kwargs.get("API密钥", "") or "").strip()

        if not api_key:
            _log("GPT-Image-2: API密钥为空")
            return (_blank_image(1024),)

        if not prompt:
            _log("GPT-Image-2: 提示词为空")
            return (_blank_image(1024),)

        # 收集参考图
        refs: List[torch.Tensor] = []
        for i in range(1, 15):
            img = kwargs.get(f"参考图{i}")
            if isinstance(img, torch.Tensor):
                refs.append(img)

        # 比例为"自动"时,如果有参考图,根据第一张参考图的宽高比自动匹配
        if aspect_ratio in {"自动", "auto", ""}:
            if refs:
                best = _choose_best_aspect_ratio_from_options(refs[0], GPT_IMAGE2_ASPECT_RATIO_OPTIONS)
                if best and best not in {"自动"}:
                    aspect_ratio = best
                    _log(f"GPT-Image-2 auto ratio from reference image: {aspect_ratio}")

        # 根据比例 + 分辨率计算实际尺寸
        size = _gpt_image2_resolve_size(aspect_ratio, resolution)

        # 判断走 generations 还是 edits
        if refs:
            return self._run_edits(prompt, size, aspect_ratio, resolution, quality_value, output_format, seed, api_key, refs, max_wait)
        else:
            return self._run_generations(prompt, size, aspect_ratio, resolution, quality_value, output_format, seed, api_key, max_wait)

    def _run_generations(self, prompt, size, aspect_ratio, resolution, quality, output_format, seed, api_key, max_wait=600):
        """文生图：POST /v1/images/generations"""
        payload: Dict[str, Any] = {
            "model": "gpt-image-2",
            "prompt": prompt,
            "size": size,
            "ratio": aspect_ratio,
            "image_size": resolution,
            "n": 1,
            "quality": quality,
            "output_format": output_format,
        }

        _log(f"GPT-Image-2 generations: size={size}, quality={quality}, format={output_format}")

        # 走原生异步图片任务
        result = self._submit_and_poll(
            path="/v1/images/generations",
            payload=payload,
            api_key=api_key,
            max_wait=max_wait,
        )
        return result

    def _run_edits(self, prompt, size, aspect_ratio, resolution, quality, output_format, seed, api_key, refs, max_wait=600):
        """图生图：POST /v1/images/edits（官方 multipart/form-data）。"""
        form_data, files = _build_gpt_image2_edit_multipart(prompt, size, aspect_ratio, resolution, quality, output_format, refs)

        _log(f"GPT-Image-2 edits: size={size}, quality={quality}, format={output_format}, refs={len(refs)}")

        # 走原生异步图片任务
        result = self._submit_multipart_and_poll(
            path="/v1/images/edits",
            form_data=form_data,
            files=files,
            api_key=api_key,
            max_wait=max_wait,
        )
        return result

    def _submit_multipart_and_poll(
        self,
        path: str,
        form_data: Dict[str, str],
        files: List[Tuple[str, Tuple[str, bytes, str]]],
        api_key: str,
        max_wait: int = 600,
    ):
        """提交 multipart 官方图片任务,轮询拿结果。"""
        submit_url = OPENAI_API_ROOT.rstrip("/") + path

        try:
            resp = _http_post(
                submit_url,
                headers=_make_multipart_headers(api_key),
                data=form_data,
                files=files,
                timeout=(120, 600),
            )
        except Exception as exc:
            _log(f"GPT-Image-2 submit exception: {exc}")
            return (_blank_image(1024),)

        return self._handle_submit_response(resp, api_key, max_wait)

    def _submit_and_poll(
        self,
        path: str,
        payload: Optional[Dict[str, Any]],
        api_key: str,
        max_wait: int = 600,
    ):
        """提交到原生图片异步通道,轮询拿结果。"""
        submit_url = OPENAI_API_ROOT.rstrip("/") + path

        try:
            resp = _http_post(submit_url, headers=_make_headers(api_key), json=payload or {}, timeout=(120, 600))
        except Exception as exc:
            _log(f"GPT-Image-2 submit exception: {exc}")
            return (_blank_image(1024),)

        return self._handle_submit_response(resp, api_key, max_wait)

    def _handle_submit_response(self, resp: requests.Response, api_key: str, max_wait: int = 600):
        _log(f"GPT-Image-2 submit result: http={resp.status_code}, body={resp.text[:500]}")

        if resp.status_code not in (200, 201, 202):
            _log(f"GPT-Image-2 submit failed: HTTP {resp.status_code}")
            return (_blank_image(1024),)

        try:
            submit_body = resp.json()
        except Exception:
            _log(f"GPT-Image-2 submit non-json: {resp.text[:300]}")
            return (_blank_image(1024),)

        # 1) 兼容极少数直接返回图片的上游。
        if isinstance(submit_body, dict) and submit_body.get("data"):
            data_list = submit_body.get("data", [])
            if isinstance(data_list, list) and data_list:
                item = data_list[0]
                # b64_json
                b64 = item.get("b64_json")
                if b64:
                    tensor = _decode_base64_image_to_tensor(b64)
                    if tensor is not None:
                        _log("GPT-Image-2: got direct b64_json response")
                        return (tensor,)
                # url
                url = item.get("url")
                if url:
                    try:
                        tensor = _download_image_to_tensor(url)
                        _log(f"GPT-Image-2: got direct url response: {url}")
                        return (tensor,)
                    except Exception as exc:
                        _log(f"GPT-Image-2: download direct url failed: {exc}")

        # 2) 原生图片异步任务：顶层 id/task_id。
        task_id, query_url = _extract_native_image_task_info(submit_body)
        if not task_id and not query_url:
            task_id, query_url = _extract_async_task_info_from_chat_response(submit_body)

        if not task_id and not query_url:
            _log(f"GPT-Image-2: no task_id in response: {json.dumps(submit_body, ensure_ascii=False)[:400]}")
            return (_blank_image(1024),)

        _log(f"GPT-Image-2 task created: {task_id}, polling (max_wait={max_wait}s)...")

        # 3) 轮询（按 max_wait 计算最大轮询次数,间隔 3 秒）
        max_attempts = max(1, max_wait // 3)
        tensor, err = _finalize_gemini_task_from_query(api_key, task_id, query_url, max_attempts=max_attempts)
        if tensor is not None:
            hw = _get_image_hw(tensor)
            if hw:
                _log(f"GPT-Image-2 output size: {hw[1]}x{hw[0]} (w x h)")
            return (tensor,)

        _log(f"GPT-Image-2 poll failed: {err}")
        return (_blank_image(1024),)


NODE_CLASS_MAPPINGS = {
    "KRLLMNode": KRLLMNode,
    "KRGPTLanguageNode": KRGPTLanguageNode,
    "KRGeminiImageNode": KRGeminiImageNode,
    "KRGeminiImageAsyncSubmitNode": KRGeminiImageAsyncSubmitNode,
    "KRGeminiImageAsyncFetchNode": KRGeminiImageAsyncFetchNode,
    "KROpenAIImageAsyncSubmitNode": KRGPTImage2AsyncSubmitNode,
    "KROpenAIImageAsyncFetchNode": KRGPTImage2AsyncFetchNode,
    "KRGPTImage2Node": KRGPTImage2Node,
    "KRVeoImageToVideoNode": KRVeoImageToVideoNode,
    "KRVeoKeyframeVideoNode": KRVeoKeyframeVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KRLLMNode": "KR-\u8bed\u8a00\u5927\u6a21\u578b",
    "KRGPTLanguageNode": "KR-GPT\u8bed\u8a00\u6a21\u578b",
    "KRGeminiImageNode": "KR-Gemini\u751f\u56fe",
    "KRGeminiImageAsyncSubmitNode": "KR-Gemini异步提交",
    "KRGeminiImageAsyncFetchNode": "KR-Gemini异步获取",
    "KROpenAIImageAsyncSubmitNode": "KR-GPT-Image-2异步提交",
    "KROpenAIImageAsyncFetchNode": "KR-GPT-Image-2异步获取",
    "KRGPTImage2Node": "KR-GPT-Image-2\u751f\u56fe",
    "KRVeoImageToVideoNode": "KR-Veo图生视频",
    "KRVeoKeyframeVideoNode": "KR-Veo首尾帧视频",
}
