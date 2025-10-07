#!/usr/bin/env python3
import os
import io
import logging
import base64
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from flask import Flask, request, render_template_string, jsonify
from pinecone import Pinecone
from werkzeug.exceptions import RequestEntityTooLarge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("advanced_image_search")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "paris")
AVAILABLE_NAMESPACES = [f"750{i:02d}" for i in range(1, 21)]

# ---- Default Top-L (used only if mode not provided) ----
def _get_topL_default() -> int:
    try:
        L = int(os.getenv("TOPL_L", "64"))
    except Exception:
        L = 64
    return max(1, min(256, L))

TOPL_L = _get_topL_default()

# ---- Mode presets (fixed Top-L, no adaptive) ----
MODE_PRESETS = {
    "full":   {"alpha": 0.75, "L_top": 144, "one_way": False, "windowed": False, "q_grid": 16},
    "mid":    {"alpha": 0.50, "L_top":  49, "one_way": False, "windowed": False, "q_grid":  8},
    "detail": {"alpha": 0.20, "L_top":  12, "one_way": True,  "windowed": False, "q_grid":  4},
}

@dataclass
class SearchResult:
    address: str
    final_score: float
    coarse_score: float
    coarse_channel: str  # kept in pipeline, hidden in UI
    gem_score: float
    cls_score: float
    location_folder: str
    image_id: str
    street_view_src: str  # NEW: Street View iframe URL (no API key)

# Global variables for lazy loading
_pinecone_client = None
_pinecone_index = None
_dinov2_model = None
_device = None
_transform = None
_proj_matrix = None

def get_pinecone_client():
    global _pinecone_client, _pinecone_index
    if _pinecone_client is None:
        try:
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY environment variable is not set")
            logger.info("Initializing Pinecone client...")
            _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            _pinecone_index = _pinecone_client.Index(PINECONE_INDEX_NAME)
            _ = _pinecone_index.describe_index_stats()
            logger.info("Pinecone connected successfully.")
        except Exception as e:
            logger.error(f"Pinecone init failed: {e}")
            _pinecone_client = None
            _pinecone_index = None
    return _pinecone_client, _pinecone_index

# === Query preproc ALIGNED WITH DATABASE ===
def get_dinov2_model():
    global _dinov2_model, _device, _transform, _proj_matrix
    if _dinov2_model is None:
        try:
            logger.info("Loading DINOv2 model...")
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {_device}")
            _dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            _dinov2_model.to(_device).eval()
            _transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            _proj_matrix = None
            logger.info("DINOv2 model loaded successfully")
        except Exception as e:
            logger.error(f"Model init failed: {e}")
            _dinov2_model = None
            _device = None
            _transform = None
            _proj_matrix = None
    return _dinov2_model, _device, _transform, _proj_matrix

# === Helpers (same math as DB) ===
def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.norm(x, p=2, dim=-1, keepdim=True) + eps)

def _ensure_projection(d_in: int):
    global _proj_matrix, _device
    if _proj_matrix is not None and _proj_matrix.shape[0] == d_in:
        return
    gen = torch.Generator()
    gen.manual_seed(42)
    P_cpu = torch.randn(d_in, 512, generator=gen, device='cpu')
    P_cpu = F.normalize(P_cpu, p=2, dim=0)
    _proj_matrix = P_cpu.to(_device)

def _project_512_and_l2(x: torch.Tensor) -> torch.Tensor:
    global _proj_matrix
    d = x.shape[-1]
    if d == 512:
        return _l2(x)
    if d < 512:
        pad = torch.zeros(x.shape[0], 512 - d, device=x.device, dtype=x.dtype)
        y = torch.cat([x, pad], dim=-1)
        return _l2(y)
    _ensure_projection(d)
    y = x @ _proj_matrix
    return _l2(y)

def _gem_over_tokens(tokens_hwD: torch.Tensor, p: float = 3.0) -> torch.Tensor:
    eps = 1e-6
    x = tokens_hwD.clamp_min(eps).pow(p)
    m = x.mean(dim=1)
    return m.pow(1.0 / p)

def _clean(v: torch.Tensor) -> torch.Tensor:
    v_np = v.detach().cpu().numpy()
    if np.any(np.isnan(v_np)) or np.any(np.isinf(v_np)):
        v_np = np.nan_to_num(v_np, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(v_np).to(v.device)
    return v

def _extract_features(image_tensor: torch.Tensor) -> torch.Tensor:
    global _dinov2_model, _device
    with torch.no_grad():
        out = _dinov2_model.forward_features(image_tensor)
        if isinstance(out, dict):
            patch = None
            for k in ["x_norm_patchtokens", "x_norm_patch_tokens", "patch_tokens", "tokens"]:
                if k in out:
                    patch = out[k]; break
            cls = None
            for k in ["x_norm_clstoken", "cls_token"]:
                if k in out:
                    cls = out[k]; break
            if patch is None or cls is None:
                raise RuntimeError(f"Unexpected DinoV2 outputs: {list(out.keys())}")
            if cls.dim() == 2:
                cls = cls.unsqueeze(1)
            tokens = torch.cat([cls.to(_device), patch.to(_device)], dim=1)
            return tokens
        if isinstance(out, torch.Tensor):
            return out.to(_device)
        raise RuntimeError("Unsupported forward_features output type")

# === Query vectorization aligned with DB ===
def vectorize_query_image_like_database(image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    model, device, transform, _ = get_dinov2_model()
    if model is None:
        raise RuntimeError("DINOv2 model not loaded")
    t = transform(image).unsqueeze(0).to(device)
    tokens = _extract_features(t)
    cls_tok, grid = tokens[:, :1, :], tokens[:, 1:, :]
    v_gem = _gem_over_tokens(grid)
    v_gem = _project_512_and_l2(v_gem).squeeze(0)
    v_gem = _clean(v_gem)
    v_cls = _project_512_and_l2(cls_tok.squeeze(1)).squeeze(0)
    v_cls = _clean(v_cls)
    return v_gem, v_cls

def vectorize_query_tokens_512(image: Image.Image) -> torch.Tensor:
    """Legacy function - uses full 16x16 grid"""
    model, device, transform, _ = get_dinov2_model()
    t = transform(image).unsqueeze(0).to(device)
    tokens = _extract_features(t)
    grid = tokens[:, 1:, :].squeeze(0)
    q_tokens_512 = _project_512_and_l2(grid)
    q_tokens_512 = _clean(q_tokens_512).detach().cpu()
    return q_tokens_512

# === Multi-scale query vectorization ===
def _pool_tokens_grid(grid_tokens: torch.Tensor, H_out: int, W_out: int, method: str = "avg", p: float = 3.0) -> torch.Tensor:
    """
    Pool a 16x16 grid of tokens to a smaller HxW grid.
    grid_tokens: (N_tokens=256, D) with H=W=16 (at 224x224)
    Returns: (H_out*W_out, D)
    """
    device = grid_tokens.device
    D = grid_tokens.shape[-1]
    # reshape (256, D) -> (16, 16, D) -> (1, D, 16, 16)
    hwD = grid_tokens.view(16, 16, D).permute(2, 0, 1).unsqueeze(0)  # (1, D, 16, 16)
    
    if method == "gem":
        eps = 1e-6
        x = torch.clamp(hwD, min=eps).pow(p)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (H_out, W_out))
        x = x.pow(1.0 / p)
    else:  # avg
        x = torch.nn.functional.adaptive_avg_pool2d(hwD, (H_out, W_out))
    
    # (1, D, H_out, W_out) -> (H_out, W_out, D) -> (H_out*W_out, D)
    x = x.squeeze(0).permute(1, 2, 0).contiguous().view(H_out * W_out, D)
    return x.to(device)

def vectorize_query_tokens_512_with_grid(image: Image.Image, q_grid: int) -> torch.Tensor:
    """
    Vectorize query image with adaptive grid size for multi-scale matching.
    Keeps 224x224 resolution but pools tokens to q_grid x q_grid.
    
    q_grid: 16 (full, 256 tokens), 8 (mid, 64 tokens), or 4 (detail, 16 tokens)
    """
    # Keep 224x224 input size
    model, device, transform, _ = get_dinov2_model()
    t = transform(image).unsqueeze(0).to(device)
    tokens = _extract_features(t)  # (1, 1+256, D)
    grid = tokens[:, 1:, :].squeeze(0)  # (256, D) = 16*16
    
    # Validate and apply pooling if needed
    if q_grid not in (16, 8, 4):
        q_grid = 16
    
    if q_grid < 16:
        grid = _pool_tokens_grid(grid, q_grid, q_grid, method="avg")
    
    # Project to 512D and L2 normalize
    q_tokens_512 = _project_512_and_l2(grid)
    q_tokens_512 = _clean(q_tokens_512).detach().cpu()
    return q_tokens_512

# === Pinecone retrieval =====
def coarse_retrieval_gem(q_gem: torch.Tensor, namespace: str, k_coarse: int = 10) -> List[Dict]:
    _, pinecone_index = get_pinecone_client()
    if pinecone_index is None:
        return []
    try:
        res = pinecone_index.query(
            vector=q_gem.detach().cpu().to(torch.float32).numpy().tolist(),
            top_k=k_coarse,
            namespace=namespace,
            filter={"vector_type": "global_gem"},
            include_metadata=True
        )
        matches = res.matches or []
        candidates = []
        for match in matches:
            if match.metadata and "location_folder" in match.metadata:
                uid = match.id.split("::")[0]
                candidates.append({
                    "id": match.id,
                    "uid": uid,
                    "score": float(match.score),
                    "metadata": match.metadata,
                    "result_type": "global_gem"
                })
        return candidates
    except Exception as e:
        logging.warning(f"GeM coarse retrieval failed: {e}")
        return []

def coarse_retrieval_cls(q_cls: torch.Tensor, namespace: str, k_coarse: int = 10) -> List[Dict]:
    _, pinecone_index = get_pinecone_client()
    if pinecone_index is None:
        return []
    try:
        res = pinecone_index.query(
            vector=q_cls.detach().cpu().to(torch.float32).numpy().tolist(),
            top_k=k_coarse,
            namespace=namespace,
            filter={"vector_type": "global_cls"},
            include_metadata=True
        )
        matches = res.matches or []
        candidates = []
        for match in matches:
            if match.metadata and "location_folder" in match.metadata:
                uid = match.id.split("::")[0]
                candidates.append({
                    "id": match.id,
                    "uid": uid,
                    "score": float(match.score),
                    "metadata": match.metadata,
                    "result_type": "global_cls"
                })
        return candidates
    except Exception as e:
        logging.warning(f"CLS coarse retrieval failed: {e}")
        return []

def fetch_candidate_tokens_512(pinecone_index, namespace: str, uid: str) -> Optional[np.ndarray]:
    ids = [f"{uid}::t::{r}_{c}" for r in range(16) for c in range(16)]
    try:
        fetched = pinecone_index.fetch(ids=ids, namespace=namespace)
        vecs = fetched.get("vectors", {})
        if len(vecs) != 256:
            logging.warning(f"UID {uid}: expected 256 tokens, got {len(vecs)}")
            return None
        tokens = np.zeros((256, 512), dtype=np.float32)
        for r in range(16):
            for c in range(16):
                vid = f"{uid}::t::{r}_{c}"
                v = vecs[vid]["values"]
                tokens[r*16 + c] = np.asarray(v, dtype=np.float32)
        norms = np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8
        tokens = tokens / norms
        return tokens
    except Exception as e:
        logging.warning(f"fetch_candidate_tokens_512 failed for {uid}: {e}")
        return None

def _safe_l2(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def batch_fetch_global_vectors(pinecone_index, namespace: str, uids: List[str]) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    ids = []
    for uid in uids:
        ids.append(f"{uid}::g_gem")
        ids.append(f"{uid}::g_cls")
    out: Dict[str, Dict[str, Optional[np.ndarray]]] = {uid: {"gem": None, "cls": None} for uid in uids}
    try:
        fetched = pinecone_index.fetch(ids=ids, namespace=namespace)
        vecs = fetched.get("vectors", {})
        for vid, payload in vecs.items():
            if "::g_gem" in vid:
                uid = vid.split("::")[0]
                vec = np.asarray(payload["values"], dtype=np.float32)
                out[uid]["gem"] = _safe_l2(vec)
            elif "::g_cls" in vid:
                uid = vid.split("::")[0]
                vec = np.asarray(payload["values"], dtype=np.float32)
                out[uid]["cls"] = _safe_l2(vec)
    except Exception as e:
        logging.warning(f"batch_fetch_global_vectors failed: {e}")
    return out

def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    if not np.isclose(np.linalg.norm(a), 1.0, atol=1e-3):
        a = _safe_l2(a)
    if not np.isclose(np.linalg.norm(b), 1.0, atol=1e-3):
        b = _safe_l2(b)
    return float(np.dot(a, b))

def _mean_of_topk(arr: np.ndarray, k: int) -> float:
    k = max(1, min(arr.size, k))
    idx = np.argpartition(arr, -k)[-k:]
    return float(arr[idx].mean())

def local_chamfer_score_topL(q_tokens_512: torch.Tensor, cand_tokens_512: np.ndarray, L: int = TOPL_L) -> float:
    Q = q_tokens_512.numpy()
    C = cand_tokens_512
    S = Q @ C.T
    row_max = S.max(axis=1)
    col_max = S.max(axis=0)
    L = max(1, min(256, int(L)))
    row_topL_mean = _mean_of_topk(row_max, L)
    col_topL_mean = _mean_of_topk(col_max, L)
    return float(0.5 * (row_topL_mean + col_topL_mean))

def local_chamfer_one_way_topL(q_tokens_512: torch.Tensor, cand_tokens_512: np.ndarray, L: int) -> float:
    Q = q_tokens_512.numpy()
    C = cand_tokens_512
    S = Q @ C.T
    row_max = S.max(axis=1)
    L = max(1, min(256, int(L)))
    return _mean_of_topk(row_max, L)

# ---- Simplified multi-scale crops for detail mode ----
def _generate_query_crops_fast(img: Image.Image) -> List[Image.Image]:
    scales = [0.7, 1.0]
    crops = []
    w, h = img.size
    smin = min(w, h)
    for s in scales:
        cw = ch = int(round(s * smin))
        if cw < 48 or ch < 48:
            continue
        x = (w - cw) // 2
        y = (h - ch) // 2
        crops.append(img.crop((x, y, x + cw, y + ch)))
    crops.append(img)
    return crops[:3]

def best_local_score_over_crops(img: Image.Image, cand_tokens_512: np.ndarray, L: int, one_way: bool) -> float:
    best = -1.0
    crops = _generate_query_crops_fast(img)
    for crop in crops:
        q_tokens = vectorize_query_tokens_512(crop)
        s = (local_chamfer_one_way_topL(q_tokens, cand_tokens_512, L)
             if one_way else local_chamfer_score_topL(q_tokens, cand_tokens_512, L))
        if s > best:
            best = s
    return best

# --- Helper: lat/long + Street View embed ---
_LAT_LONG_RE = re.compile(r'_lat_([-\d\.]+)_long_([-\d\.]+)', re.IGNORECASE)

def _latlon_from_location_folder(location_folder: str) -> Optional[Tuple[str, str]]:
    if not location_folder:
        return None
    m = _LAT_LONG_RE.search(location_folder)
    if not m:
        return None
    return m.group(1), m.group(2)

def _street_view_src_from_latlon(lat: str, lon: str) -> str:
    # Google Street View embed (no key)
    # Example: https://maps.google.com/maps?q=&layer=c&cbll=48.8382142,2.35597031&cbp=12,0,,0,0&output=svembed
    return f"https://maps.google.com/maps?q=&layer=c&cbll={lat},{lon}&cbp=12,0,,0,0&output=svembed"

# === Pipeline ===
def database_matching_search_pipeline(
    image: Image.Image,
    namespace: str,
    k_coarse: int = 20,
    alpha: float = 0.5,
    L_top: int = TOPL_L,
    mode: str = "mid"
) -> List[SearchResult]:
    preset = MODE_PRESETS.get(mode, MODE_PRESETS["mid"])
    alpha = preset["alpha"]
    L_top = preset["L_top"]
    one_way = preset["one_way"]
    windowed = preset["windowed"]
    q_grid = preset.get("q_grid", 16)

    pc_client, pinecone_index = get_pinecone_client()
    if pinecone_index is None:
        return []

    # Global vectors (unchanged): same for all modes
    q_gem, q_cls = vectorize_query_image_like_database(image)

    gem_candidates = coarse_retrieval_gem(q_gem, namespace, k_coarse=k_coarse)
    cls_candidates = coarse_retrieval_cls(q_cls, namespace, k_coarse=k_coarse)

    by_uid: Dict[str, Dict] = {}
    for cand in gem_candidates:
        uid = cand["uid"]
        by_uid.setdefault(uid, {"uid": uid, "gem_score": 0.0, "cls_score": 0.0, "metadata": cand.get("metadata", {})})
        by_uid[uid]["gem_score"] = max(by_uid[uid]["gem_score"], float(cand["score"]))

    for cand in cls_candidates:
        uid = cand["uid"]
        by_uid.setdefault(uid, {"uid": uid, "gem_score": 0.0, "cls_score": 0.0, "metadata": cand.get("metadata", {})})
        by_uid[uid]["cls_score"] = max(by_uid[uid]["cls_score"], float(cand["score"]))

    if not by_uid:
        return []

    uids = list(by_uid.keys())
    globals_map = batch_fetch_global_vectors(pinecone_index, namespace, uids)

    q_gem_np = q_gem.detach().cpu().to(torch.float32).numpy()
    q_cls_np = q_cls.detach().cpu().to(torch.float32).numpy()

    for uid in uids:
        g = globals_map.get(uid, {})
        gem_vec = g.get("gem", None)
        cls_vec = g.get("cls", None)
        if gem_vec is not None:
            by_uid[uid]["gem_score"] = max(by_uid[uid]["gem_score"], cosine_np(q_gem_np, gem_vec))
        if cls_vec is not None:
            by_uid[uid]["cls_score"] = max(by_uid[uid]["cls_score"], cosine_np(q_cls_np, cls_vec))

    # Local vectors: use adaptive grid size based on mode
    q_tokens_512 = None if windowed else vectorize_query_tokens_512_with_grid(image, q_grid)

    ranked: List[Tuple[str, float, float, str, float, float, Dict]] = []
    for uid, rec in by_uid.items():
        gem = float(rec["gem_score"])
        cls = float(rec["cls_score"])
        coarse = gem if gem >= cls else cls
        channel = "global_gem" if gem >= cls else "global_cls"

        cand_tokens = fetch_candidate_tokens_512(pinecone_index, namespace, uid)  # Always 16x16=256 from DB
        if cand_tokens is None:
            local = coarse
        else:
            if windowed:
                local = best_local_score_over_crops(image, cand_tokens, L=L_top, one_way=one_way)
            else:
                # Chamfer works with different sizes (Q×C): query can be smaller than candidate
                local = (local_chamfer_one_way_topL(q_tokens_512, cand_tokens, L=L_top)
                         if one_way else local_chamfer_score_topL(q_tokens_512, cand_tokens, L=L_top))

        final_score = alpha * coarse + (1 - alpha) * local
        ranked.append((uid, final_score, coarse, channel, gem, cls, rec["metadata"]))

    ranked.sort(key=lambda x: x[1], reverse=True)

    results: List[SearchResult] = []
    for uid, final_score, coarse_score, channel, gem_score, cls_score, md in ranked[:5]:
        location_folder = md.get("location_folder", "")
        address = location_folder.replace("_lat_", " ").replace("_long_", " ").replace("_", " ")

        latlon = _latlon_from_location_folder(location_folder)
        if latlon:
            lat, lon = latlon
            sv_src = _street_view_src_from_latlon(lat, lon)
        else:
            sv_src = ""

        results.append(SearchResult(
            address=address,
            final_score=final_score,
            coarse_score=coarse_score,
            coarse_channel=channel,
            gem_score=gem_score,
            cls_score=cls_score,
            location_folder=location_folder,
            image_id=uid,
            street_view_src=sv_src
        ))

    return results

logger.info("Application starting with lazy loading...")

# Single Flask app (avoid double instantiation)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size

# Custom error handler for file size limit
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return render_template_string('''
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Fichier trop volumineux</title>
<style>
body { font-family: Inter, system-ui, -apple-system, Segoe UI, Arial; max-width: 720px; margin: 60px auto; padding: 0 20px; }
.card { background: #fff; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,.08); }
h2 { margin: 0 0 12px; }
.error { color: #a40000; background: #ffe6e6; padding: 14px; border-radius: 10px; text-align: left; }
a { color: #6b46ff; text-decoration: none; }
</style>
</head>
<body>
<div class="card">
<div class="error">
<h2>Fichier trop volumineux</h2>
<p>L'image envoyée dépasse 64&nbsp;Mo.</p>
<p><a href="/">← Retour</a></p>
</div>
</div>
</body>
</html>
'''), 413

# ======== UI (updated to Street View embed) ========
HTML_TEMPLATE = r'''
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Vue sur Rue — retrouvez l'adresse d'une annonce immo à partir d'une photo</title>
<link href="https://unpkg.com/cropperjs@1.6.2/dist/cropper.min.css" rel="stylesheet"/>
<style>
:root{ --violet-1:#7b5cff; /* main */ --violet-2:#a076ff; /* lighter */ --violet-3:#e9e3ff; /* very light */ --text:#1f2430; --muted:#6b7280; --card:#ffffff; --border:#ececec; }
*{box-sizing:border-box; -webkit-tap-highlight-color: transparent;}
body{ margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial; color:var(--text); background: linear-gradient(180deg, #f7f6ff 0%, #ffffff 60%); }
form{ position: relative; z-index: 1; }
.hero{ background:#ffffff; color:var(--violet-1); padding: 28px 20px 18px; border-bottom: 1px solid var(--border); }
.hero-inner{ max-width: 1100px; margin: 0 auto; display:flex; align-items:center; gap:16px; }
.hero-badge{ width:44px;height:44px;border-radius:12px;background:#f5f1ff;display:flex;align-items:center;justify-content:center; border:1px solid #e6e0ff; font-size:22px; color:var(--violet-1); }
.hero h1{ margin:0; font-size: 22px; font-weight: 800; line-height:1.25; color:var(--violet-1); }
.container{ max-width:1100px; margin: 16px auto 40px; padding: 0 20px; }
.card{ background:var(--card); border-radius:20px; box-shadow: 0 18px 60px rgba(96, 72, 255, .12); border:1px solid var(--border); padding: 22px; position: relative; z-index: 1; }
label{ font-weight:600; font-size:14px; color:#111; display:block; margin-bottom:8px; position: relative; z-index: 1; }
.stack{ display:flex; flex-direction:column; gap:12px; position: relative; z-index: 1; }
.mode-group{ display:flex; gap:10px; position: relative; z-index: 100; justify-content: flex-start; }
/* Inputs */
select, .btn{ 
    appearance:none; 
    border:1px solid var(--border); 
    border-radius:12px; 
    padding:12px 14px; 
    font-size:14px; 
    background:#fff; 
    cursor:pointer !important; 
    pointer-events: auto !important; 
    position: relative; 
    z-index: 1000 !important;
    user-select: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    touch-action: manipulation;
}
.btn-primary{ 
    background:var(--violet-1); 
    border-color:transparent; 
    color:#fff; 
    font-weight:600; 
    transition: transform .05s ease, box-shadow .2s ease; 
}
.btn-primary:hover{ 
    transform: translateY(-1px); 
    box-shadow:0 8px 18px rgba(123,92,255,.35); 
}
.btn-primary:active{ 
    transform: translateY(0px); 
}
.mode-btn{ 
    flex: 0 0 auto;
    text-align:center; 
    padding:8px 16px; 
    border-radius:999px; 
    border:1px solid var(--border); 
    background:#faf9ff; 
    cursor:pointer !important; 
    font-weight:600; 
    font-size:13px; 
    transition: all 0.2s ease; 
    pointer-events: auto !important; 
    position: relative; 
    z-index: 1000 !important;
    user-select: none;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    touch-action: manipulation;
    white-space: nowrap;
}
.mode-btn:hover{ background:#f0edff; transform: translateY(-1px); }
.mode-btn:active{ transform: translateY(0px); }
.mode-btn[data-active="true"]{ background: linear-gradient(90deg, var(--violet-1), var(--violet-2)); color:#fff; border-color:transparent; box-shadow:0 6px 14px rgba(123,92,255,.35) }
/* Drop zone */
.dropzone{ border:2px dashed #d9d6ff; background: #fbfaff; border-radius:16px; padding:24px; text-align:center; transition: border-color .2s ease, background .2s ease; position: relative; }
.dropzone.dragover{ border-color: var(--violet-1); background:#f4f1ff }
.dz-title{ font-weight:700; margin-bottom:6px }
.dz-sub{ color:var(--muted); font-size:13px; margin-bottom:12px }
.dropzone-info{ 
    margin-top: 16px; 
    padding: 12px; 
    background: #f7f4ff; 
    border-radius: 8px; 
    font-size: 12px; 
    color: #4a3f7a; 
    text-align: left; 
    line-height: 1.5;
}
.hidden-input{ display:none }
.actions{ display:flex; justify-content:center; margin-top: 18px }
/* Cropper purple theming */
.cropper-view-box { outline: 1px solid var(--violet-1); }
.cropper-line, .cropper-point { background-color: var(--violet-1); }
.cropper-face { background: transparent; }
/* Results */
.results{ margin-top:20px }
.results-title{ font-weight: 700; color:#2a235f; margin: 4px 0 12px; }
.query-image-container{ 
    margin: 16px 0 20px; 
    padding: 16px; 
    background: #f7f4ff; 
    border-radius: 14px; 
    border: 1px solid #e7e0ff;
}
.query-image-title{ 
    font-weight: 600; 
    color: #2a235f; 
    margin-bottom: 12px; 
    font-size: 14px;
}
.query-image{ 
    max-width: 300px; 
    max-height: 300px; 
    border-radius: 10px; 
    border: 2px solid var(--violet-1); 
    box-shadow: 0 4px 12px rgba(123, 92, 255, 0.2);
    display: block;
}
.results-grid{ display:grid; grid-template-columns: repeat(5, 1fr); gap:12px; align-items:stretch; }
@media (max-width: 1100px){ .results-grid{ grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); } }
.result-card{ border:1px solid var(--border); padding:0; border-radius:14px; background:#fff; overflow:hidden; display:flex; flex-direction:column; min-height:120px; }
.sv-wrap{ position:relative }
.sv-iframe{ border:0; width:100%; height:260px; display:block; background:#f4f1ff; }
/* New: rank below the map, purple, not overlaying the map */
.rank-tag{ display:inline-block; margin:10px 12px 4px; padding:6px 10px; border-radius:999px; border:1px solid #e6e0ff; background:#f7f4ff; color:var(--violet-1); font-weight:800; font-size:12px; line-height:1; align-self:flex-start; }
/* Loader overlay */
.loader-overlay{ position: fixed; inset: 0; background: rgba(255,255,255,.75); display:none; align-items:center; justify-content:center; z-index: 9999; backdrop-filter: blur(1px); pointer-events: none; }
.loader-overlay[style*="display: flex"]{ pointer-events: auto; }
.loader{ width: 64px; height: 64px; border-radius: 50%; border: 6px solid #e9e3ff; border-top-color: var(--violet-1); animation: spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
/* Info banner */
.info{ margin-top:16px; padding:14px; border-radius:14px; background:#f7f4ff; border:1px solid #e7e0ff; color:#2a235f; }
</style>
</head>
<body>
<div class="hero">
<div class="hero-inner">
<div class="hero-badge">📍</div>
<h1>Vue sur Rue — retrouvez l'adresse d'une annonce immo à partir d'une photo</h1>
</div>
</div>
<div class="container">
<div class="card">
<form id="searchForm" method="post" enctype="multipart/form-data" class="stack" novalidate>
<!-- Step 1: Arrondissement -->
<div>
<label>Étape 1 - Sélectionnez l'arrondissement</label>
<select name="namespace" required>
<option value="">Choisir…</option>
''' + ''.join([f'<option value="{ns}">{ns}</option>' for ns in AVAILABLE_NAMESPACES]) + '''
</select>
</div>
<!-- Step 2: Mode -->
<div>
<label>Étape 2 - Sélectionnez le mode</label>
<div class="mode-group" id="modeGroup">
<button type="button" class="mode-btn" data-mode="full">Façade</button>
<button type="button" class="mode-btn" data-mode="mid" data-active="true">Façade Partielle</button>
<button type="button" class="mode-btn" data-mode="detail">Élément Architectural</button>
</div>
<input type="hidden" name="mode" id="modeInput" value="mid">
</div>
<!-- Step 3: Image -->
<div>
<label>Étape 3 - Téléchargez une image et sélectionnez le cadrage d'analyse</label>
<div class="dropzone" id="dropzone">
<div class="dz-title">Glissez-déposez une image</div>
<div class="dz-sub">ou</div>
<button type="button" class="btn btn-primary" id="pickBtn" aria-controls="fileInput">Téléchargez une image</button>
<input class="hidden-input" type="file" id="fileInput" name="image" accept="image/*" required>
<div class="dropzone-info">
Pour optimiser les résultats, recadrez l'image sur l'extérieur (évitez vitres/reflets) et choisissez le mode adapté : façade entière, façade partielle ou élément architectural. Priorisez des éléments visibles depuis la rue. Évitez toitures et éléments non visibles depuis la rue par un humain.
</div>
</div>
<!-- Hidden by default: no empty preview shown -->
<div class="cropper-area" id="cropperArea" aria-live="polite" style="display:none">
<img id="preview" alt="Aperçu de l'image sélectionnée">
</div>
</div>
<!-- Hidden input for cropped JPEG -->
<input type="hidden" name="image_base64" id="imageBase64">
<div class="actions">
<button type="submit" class="btn btn-primary" id="searchBtn">Rechercher</button>
</div>
</form>
{% if error %}
<div class="info" style="background:#fff8f8;border-color:#ffd5d5;color:#8a1010;">{{ error }}</div>
{% endif %}
{% if results %}
<div class="results">
<div class="results-title">Résultats par ordre de priorité.</div>
{% if query_image %}
<div class="query-image-container">
<div class="query-image-title">Votre image de recherche :</div>
<img src="{{ query_image }}" alt="Image de recherche" class="query-image">
</div>
{% endif %}
<div class="results-grid">
{% for r in results[:5] %}
<div class="result-card">
{% set rank = loop.index %}
{% if r.street_view_src %}
<div class="sv-wrap">
<iframe class="sv-iframe" title="Street View {{ rank }}" src="{{ r.street_view_src }}" allowfullscreen loading="lazy" referrerpolicy="no-referrer-when-downgrade" ></iframe>
</div>
<!-- Rank below the map, purple -->
<div class="rank-tag">#{{ rank }}</div>
{% else %}
<div class="sv-wrap" style="pointer-events:none;opacity:.75;">
<div class="sv-iframe" role="img" aria-label="Street View indisponible"></div>
</div>
<div class="rank-tag">#{{ rank }}</div>
{% endif %}
</div>
{% endfor %}
</div>
</div>
{% endif %}
</div>
</div>
<!-- Loader overlay (appears after clicking Rechercher) -->
<div class="loader-overlay" id="loaderOverlay" aria-hidden="true" role="status">
<div class="loader" aria-label="Recherche en cours"></div>
</div>
<script src="https://unpkg.com/cropperjs@1.6.2/dist/cropper.min.js"></script>
<script>
(function() {
    'use strict';
    console.log('[INIT] Script loading...');
    
    // DEBUG: Check for overlays
    function checkForOverlays() {
        console.log('[DEBUG] Checking for overlay issues...');
        const allElements = document.querySelectorAll('*');
        const problematic = [];
        
        allElements.forEach(el => {
            const style = window.getComputedStyle(el);
            const zIndex = parseInt(style.zIndex);
            const position = style.position;
            const pointerEvents = style.pointerEvents;
            
            // Check for high z-index elements that might block
            if (zIndex > 100 && position === 'fixed' && pointerEvents !== 'none') {
                problematic.push({
                    element: el,
                    tagName: el.tagName,
                    id: el.id,
                    className: el.className,
                    zIndex: zIndex,
                    position: position,
                    pointerEvents: pointerEvents,
                    display: style.display
                });
            }
        });
        
        if (problematic.length > 0) {
            console.warn('[DEBUG] Found potentially blocking elements:', problematic);
        } else {
            console.log('[DEBUG] No obvious overlay issues found');
        }
    }
    
    function initApp() {
        console.log('[INIT] Initializing app...');
        
        // Run overlay check
        checkForOverlays();
        
        // DEBUG: Add global click listener to see what's being clicked
        document.addEventListener('click', function(e) {
            console.log('[GLOBAL CLICK]', {
                target: e.target,
                tagName: e.target.tagName,
                id: e.target.id,
                className: e.target.className,
                x: e.clientX,
                y: e.clientY
            });
        }, true); // Use capture phase
        
        // Get all elements
        const modeGroup = document.getElementById('modeGroup');
        const modeInput = document.getElementById('modeInput');
        const pickBtn = document.getElementById('pickBtn');
        const fileInput = document.getElementById('fileInput');
        const dropzone = document.getElementById('dropzone');
        const cropperArea = document.getElementById('cropperArea');
        const preview = document.getElementById('preview');
        const imageBase64 = document.getElementById('imageBase64');
        const form = document.getElementById('searchForm');
        const searchBtn = document.getElementById('searchBtn');
        const loaderOverlay = document.getElementById('loaderOverlay');
        
        // Cropper instance
        let cropper = null;
        
        // Debug: Log elements
        console.log('[DEBUG] Elements found:');
        console.log('- modeGroup:', !!modeGroup, modeGroup);
        console.log('- modeInput:', !!modeInput);
        console.log('- pickBtn:', !!pickBtn, pickBtn);
        console.log('- fileInput:', !!fileInput);
        
        // ===== MODE BUTTONS =====
        if (modeGroup && modeInput) {
            console.log('[SETUP] Setting up mode buttons...');
            const modeButtons = modeGroup.querySelectorAll('.mode-btn');
            console.log('[SETUP] Found', modeButtons.length, 'mode buttons');
            
            modeButtons.forEach(function(btn, index) {
                console.log('[SETUP] Button', index, ':', btn.textContent.trim(), 'mode:', btn.getAttribute('data-mode'));
                
                btn.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('[CLICK] Mode button clicked:', this.getAttribute('data-mode'));
                    
                    // Remove active from all
                    modeButtons.forEach(function(b) {
                        b.removeAttribute('data-active');
                    });
                    
                    // Set active on clicked
                    this.setAttribute('data-active', 'true');
                    modeInput.value = this.getAttribute('data-mode');
                    console.log('[UPDATE] Mode changed to:', modeInput.value);
                });
            });
        } else {
            console.error('[ERROR] Mode elements not found!');
        }
        
        // ===== FILE PICKER BUTTON =====
        if (pickBtn && fileInput) {
            console.log('[SETUP] Setting up file picker button...');
            
            pickBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('[CLICK] Pick button clicked');
                fileInput.click();
            });
            
            pickBtn.addEventListener('mousedown', function(e) {
                console.log('[EVENT] Pick button mousedown');
            });
        } else {
            console.error('[ERROR] File picker elements not found!');
        }
        
        // ===== FILE HANDLING =====
        function handleFile(file) {
            if (!file) {
                console.warn('[FILE] No file provided');
                return;
            }
            
            console.log('[FILE] Handling file:', file.name, file.size, 'bytes');
            
            if (file.size > 64 * 1024 * 1024) {
                alert("Fichier trop volumineux. Sélectionnez une image < 64 Mo.");
                fileInput.value = '';
                imageBase64.value = '';
                if (cropper) {
                    cropper.destroy();
                    cropper = null;
                }
                if (cropperArea) cropperArea.style.display = 'none';
                if (preview) preview.src = '';
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                if (preview) {
                    preview.src = e.target.result;
                    if (cropperArea) cropperArea.style.display = 'block';
                    if (cropper) {
                        cropper.destroy();
                        cropper = null;
                    }
                    cropper = new Cropper(preview, {
                        viewMode: 1,
                        aspectRatio: 1,
                        dragMode: 'move',
                        autoCropArea: 1.0,
                        responsive: true,
                        background: false,
                    });
                    console.log('[CROPPER] Initialized');
                }
            };
            reader.readAsDataURL(file);
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', function() {
                console.log('[EVENT] File input change');
                handleFile(this.files && this.files[0]);
            });
        }
        
        // ===== DRAG AND DROP =====
        if (dropzone) {
            ['dragenter', 'dragover'].forEach(function(evt) {
                dropzone.addEventListener(evt, function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    this.classList.add('dragover');
                });
            });
            
            ['dragleave', 'drop'].forEach(function(evt) {
                dropzone.addEventListener(evt, function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    this.classList.remove('dragover');
                });
            });
            
            dropzone.addEventListener('drop', function(e) {
                const f = e.dataTransfer.files && e.dataTransfer.files[0];
                if (f) {
                    console.log('[DROP] File dropped:', f.name);
                    const dt = new DataTransfer();
                    dt.items.add(f);
                    if (fileInput) fileInput.files = dt.files;
                    handleFile(f);
                }
            });
        }
        
        // ===== FORM SUBMISSION =====
        if (form && searchBtn && loaderOverlay) {
            form.addEventListener('submit', function(e) {
                const file = fileInput && fileInput.files && fileInput.files[0];
                if (!file) {
                    alert("Veuillez d'abord sélectionner une image.");
                    e.preventDefault();
                    return;
                }
                
                if (cropper) {
                    const canvas = cropper.getCroppedCanvas({ width: 512, height: 512 });
                    if (!canvas) {
                        alert("Impossible de recadrer cette image. Essayez-en une autre.");
                        e.preventDefault();
                        return;
                    }
                    if (imageBase64) imageBase64.value = canvas.toDataURL('image/jpeg', 0.8);
                } else {
                    if (imageBase64) imageBase64.value = '';
                }
                
                searchBtn.disabled = true;
                loaderOverlay.style.display = 'flex';
                loaderOverlay.setAttribute('aria-hidden', 'false');
                console.log('[SUBMIT] Form submitted');
            });
        }
        
        console.log('[INIT] JavaScript initialized successfully');
    }
    
    // Run immediately if DOM is ready, otherwise wait
    if (document.readyState === 'loading') {
        console.log('[INIT] Waiting for DOM...');
        document.addEventListener('DOMContentLoaded', initApp);
    } else {
        console.log('[INIT] DOM already ready, running now');
        initApp();
    }
})();
</script>
</body>
</html>
'''

def _decode_data_url(data_url: str) -> Optional[bytes]:
    if not data_url:
        return None
    m = re.match(r'^data:image/(png|jpeg|jpg);base64,(.*)$', data_url, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    b64 = m.group(2)
    return base64.b64decode(b64)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    results = None
    query_image = None

    if request.method == 'POST':
        pc_client, pinecone_index = get_pinecone_client()
        model, device, transform, proj_matrix = get_dinov2_model()
        if not pinecone_index or not model:
            error = "Service non initialisé correctement. Vérifiez les logs."
        else:
            try:
                ns = request.form.get('namespace')
                mode = request.form.get('mode', 'mid')
                if not ns or not mode:
                    error = "Veuillez choisir un arrondissement et un mode."
                else:
                    img_obj = None
                    # Prefer client-side cropped data URL if provided
                    data_url = request.form.get('image_base64', '').strip()
                    if data_url:
                        raw = _decode_data_url(data_url)
                        if raw:
                            img_obj = Image.open(io.BytesIO(raw))
                            # Store the cropped image as base64 for display
                            query_image = data_url
                        else:
                            error = "Image recadrée invalide."

                    # Fallback to raw file if no cropped data provided
                    if img_obj is None and 'image' in request.files:
                        file = request.files['image']
                        if file and file.filename:
                            file_bytes = file.read()
                            img_obj = Image.open(io.BytesIO(file_bytes))
                            # Convert to base64 for display
                            buffered = io.BytesIO()
                            img_obj.save(buffered, format="JPEG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode()
                            query_image = f"data:image/jpeg;base64,{img_base64}"

                    if img_obj is None:
                        error = "Aucune image fournie."
                    else:
                        results = database_matching_search_pipeline(img_obj, ns, mode=mode)
                        results = results[:5] if results else []
                        if not results:
                            error = f"Aucun résultat trouvé dans {ns}."

            except Exception as e:
                logger.exception(f"Request error: {e}")
                error = f"Erreur: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        error=error,
        results=results,
        query_image=query_image,
        AVAILABLE_NAMESPACES=AVAILABLE_NAMESPACES
    )

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/stats')
def stats():
    pc_client, pinecone_index = get_pinecone_client()
    if not pinecone_index:
        return jsonify({"error": "Index not initialized"}), 500
    try:
        s = pinecone_index.describe_index_stats()
        return jsonify(s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/<namespace>')
def debug_namespace(namespace):
    pc_client, pinecone_index = get_pinecone_client()
    if not pinecone_index:
        return jsonify({"error": "Index not initialized"}), 500

    results = {
        "namespace": namespace,
        "total_vectors": 0,
        "vector_types": {},
        "sample_vectors": [],
        "errors": []
    }

    try:
        stats = pinecone_index.describe_index_stats()
        results["total_vectors"] = stats.get("total_vector_count", 0)
        results["namespaces"] = list(stats.get("namespaces", {}).keys())

        test_vector = [0.1] * 512
        for vector_type in ["global_gem", "global_cls"]:
            try:
                res = pinecone_index.query(
                    vector=test_vector,
                    top_k=5,
                    namespace=namespace,
                    filter={"vector_type": vector_type},
                    include_metadata=True
                )
                matches = res.matches or []
                results["vector_types"][vector_type] = len(matches)
                if matches:
                    results["sample_vectors"].append({
                        "type": vector_type,
                        "sample_id": matches[0].id,
                        "sample_metadata": matches[0].metadata
                    })
            except Exception as e:
                results["errors"].append(f"{vector_type}: {str(e)}")

        try:
            res = pinecone_index.query(
                vector=test_vector,
                top_k=5,
                namespace=namespace,
                include_metadata=True
            )
            matches = res.matches or []
            results["vector_types"]["no_filter"] = len(matches)
            if matches:
                results["sample_vectors"].append({
                    "type": "no_filter",
                    "sample_id": matches[0].id,
                    "sample_metadata": matches[0].metadata
                })
        except Exception as e:
            results["errors"].append(f"no_filter: {str(e)}")

    except Exception as e:
        results["errors"].append(f"Connection error: {str(e)}")

    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
