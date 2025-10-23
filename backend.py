#!/usr/bin/env python3
import os
import io
import logging
import base64
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pinecone import Pinecone
from werkzeug.exceptions import RequestEntityTooLarge

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

# Import authentication modules
from auth import initialize_firebase, require_approved_user, require_admin, get_current_user
from user_manager import user_manager, UserStatus, UserRole
from admin_auth import authenticate_admin, verify_admin_token
from functools import wraps
import stripe

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("advanced_image_search")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY")
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
    for uid, final_score, coarse_score, channel, gem_score, cls_score, md in ranked[:3]:
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

# Initialize Firebase Auth
firebase_initialized = initialize_firebase()
if not firebase_initialized:
    logger.warning("Firebase Auth not initialized - authentication will be disabled")

# Admin token decorator
def require_admin_token(f):
    """Decorator to require admin token authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization token is missing or invalid'}), 401
        
        token = auth_header[7:]
        payload = verify_admin_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired admin token'}), 401
        
        request.admin_user = payload
        return f(*args, **kwargs)
    return decorated_function

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size

def _decode_data_url(data_url: str) -> Optional[bytes]:
    if not data_url:
        return None
    m = re.match(r'^data:image/(png|jpeg|jpg);base64,(.*)$', data_url, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    b64 = m.group(2)
    return base64.b64decode(b64)

# Serve the frontend HTML file
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'frontend.html')

# Serve the admin panel
@app.route('/admin')
def serve_admin():
    return send_from_directory('.', 'admin.html')

# Serve the password-based admin panel
@app.route('/admin-password')
def serve_admin_password():
    return send_from_directory('.', 'admin_password.html')

# Serve the approval page
@app.route('/approve')
def serve_approve():
    return send_from_directory('.', 'approve.html')

# API endpoint for image search
@app.route('/api/search', methods=['POST'])
@require_approved_user
def search():
    try:
        # Get current user
        current_user = get_current_user()
        if not current_user:
            return jsonify({"error": "User not found"}), 400
        
        # Check usage limits
        usage_check = user_manager.check_usage_limits(current_user.email)
        if not usage_check["can_search"]:
            return jsonify({
                "error": "Usage limit reached",
                "reason": usage_check["reason"],
                "message": usage_check["message"]
            }), 403
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        namespace = data.get('namespace')
        mode = data.get('mode', 'mid')
        image_base64 = data.get('image_base64', '').strip()
        
        if not namespace or not mode:
            return jsonify({"error": "Namespace and mode are required"}), 400
        
        if not image_base64:
            return jsonify({"error": "Image is required"}), 400
        
        # Decode the base64 image
        raw = _decode_data_url(image_base64)
        if not raw:
            return jsonify({"error": "Invalid image data"}), 400
        
        img_obj = Image.open(io.BytesIO(raw))
        
        # Check if Pinecone is properly configured
        pc_client, pinecone_index = get_pinecone_client()
        if not pinecone_index:
            return jsonify({
                "success": False,
                "error": "Pinecone not configured. Please set PINECONE_API_KEY environment variable.",
                "query_image": image_base64
            }), 500
        
        # Perform search
        results = database_matching_search_pipeline(img_obj, namespace, mode=mode)
        results = results[:3] if results else []
        
        # Convert results to JSON-serializable format
        results_data = []
        for result in results:
            results_data.append({
                "address": result.address,
                "final_score": result.final_score,
                "coarse_score": result.coarse_score,
                "coarse_channel": result.coarse_channel,
                "gem_score": result.gem_score,
                "cls_score": result.cls_score,
                "location_folder": result.location_folder,
                "image_id": result.image_id,
                "street_view_src": result.street_view_src
            })
        
        # Increment usage count after successful search
        user_manager.increment_usage(current_user.email)
        
        return jsonify({
            "success": True,
            "results": results_data,
            "query_image": image_base64
        })
        
    except Exception as e:
        logger.exception(f"Search error: {e}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

# API endpoint to get available namespaces
@app.route('/api/namespaces', methods=['GET'])
@require_approved_user
def get_namespaces():
    return jsonify({"namespaces": AVAILABLE_NAMESPACES})

# Stripe payment endpoints
@app.route('/api/payment/create-checkout-session', methods=['POST'])
@require_approved_user
def create_checkout_session():
    """Create Stripe checkout session for subscription"""
    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({"error": "User not found"}), 400
        
        # Create or get Stripe customer
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                metadata={'user_email': current_user.email}
            )
            user_manager.set_stripe_customer(current_user.email, customer.id)
            current_user.stripe_customer_id = customer.id
        
        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            customer=current_user.stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'eur',
                    'product_data': {
                        'name': 'Vue Sur Rue - Monthly Subscription',
                        'description': '2 image searches per month',
                    },
                    'unit_amount': 999,  # €9.99 in cents
                    'recurring': {
                        'interval': 'month',
                    },
                },
                'quantity': 1,
            }],
            mode='subscription',
            success_url=request.url_root + '?payment=success',
            cancel_url=request.url_root + '?payment=cancelled',
            metadata={'user_email': current_user.email}
        )
        
        return jsonify({
            'checkout_url': checkout_session.url,
            'session_id': checkout_session.id
        })
        
    except Exception as e:
        logger.exception(f"Stripe checkout error: {e}")
        return jsonify({"error": f"Payment setup failed: {str(e)}"}), 500

@app.route('/api/payment/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError:
        return jsonify({"error": "Invalid signature"}), 400
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_email = session['metadata']['user_email']
        user_manager.set_subscription_status(user_email, 'active')
        logger.info(f"Subscription activated for {user_email}")
    
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        customer_id = subscription['customer']
        # Find user by customer ID and deactivate subscription
        for user in user_manager.get_all_users():
            if user.stripe_customer_id == customer_id:
                user_manager.set_subscription_status(user.email, 'canceled')
                logger.info(f"Subscription canceled for {user.email}")
                break
    
    return jsonify({"status": "success"})

@app.route('/api/usage/check', methods=['GET'])
@require_approved_user
def check_usage():
    """Check current usage status"""
    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({"error": "User not found"}), 400
        
        usage_check = user_manager.check_usage_limits(current_user.email)
        
        return jsonify({
            "can_search": usage_check["can_search"],
            "reason": usage_check.get("reason"),
            "message": usage_check.get("message"),
            "free_usage_count": current_user.free_usage_count,
            "paid_usage_count": current_user.paid_usage_count,
            "subscription_status": current_user.subscription_status
        })
        
    except Exception as e:
        logger.exception(f"Usage check error: {e}")
        return jsonify({"error": f"Usage check failed: {str(e)}"}), 500

# Admin password authentication endpoint
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Password-based admin login"""
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Email and password are required"}), 400
        
        email = data['email']
        password = data['password']
        
        token = authenticate_admin(email, password)
        if token:
            return jsonify({
                'success': True,
                'token': token,
                'user': {
                    'email': email,
                    'role': 'admin'
                }
            })
        else:
            return jsonify({"error": "Invalid credentials"}), 401
            
    except Exception as e:
        logger.exception(f"Admin login error: {e}")
        return jsonify({"error": f"Login failed: {str(e)}"}), 500

# Check user status without authentication
@app.route('/api/auth/check-status', methods=['POST'])
def check_user_status():
    """Check if a user is approved without requiring authentication"""
    try:
        data = request.get_json()
        if not data or 'email' not in data:
            return jsonify({"error": "Email is required"}), 400
        
        email = data['email']
        user = user_manager.get_user(email)
        
        if user:
            return jsonify({
                'exists': True,
                'status': user.status.value,
                'role': user.role.value
            })
        else:
            return jsonify({
                'exists': False,
                'status': 'pending',
                'role': 'user'
            })
            
    except Exception as e:
        logger.exception(f"Check user status error: {e}")
        return jsonify({"error": f"Failed to check user status: {str(e)}"}), 500

# Authentication endpoints
@app.route('/api/auth/verify', methods=['POST'])
def verify_auth():
    """Verify user authentication and return user status"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No authorization header'}), 401
        
        token = auth_header[7:]
        from auth import get_user_from_token
        user_info = get_user_from_token(token)
        
        if not user_info:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        # Check if user exists in our system
        user = user_manager.get_user(user_info['email'])
        if not user:
            # Create new user if they don't exist
            user = user_manager.create_user(
                email=user_info['email'],
                firebase_uid=user_info['uid']
            )
        else:
            # Update Firebase UID if it's different (for default admin)
            if user.firebase_uid != user_info['uid']:
                user.firebase_uid = user_info['uid']
                user.updated_at = datetime.utcnow().isoformat()
                user_manager.save_users()
        
        return jsonify({
            'authenticated': True,
            'user': {
                'email': user.email,
                'status': user.status.value,
                'role': user.role.value,
                'created_at': user.created_at,
                'last_login': user.last_login
            }
        })
        
    except Exception as e:
        logger.exception(f"Auth verification error: {e}")
        return jsonify({"error": f"Auth verification failed: {str(e)}"}), 500

# Admin endpoints
@app.route('/api/admin/users', methods=['GET'])
@require_admin_token
def get_all_users():
    """Get all users (admin only)"""
    try:
        users = user_manager.get_all_users()
        users_data = []
        for user in users:
            users_data.append({
                'email': user.email,
                'status': user.status.value,
                'role': user.role.value,
                'created_at': user.created_at,
                'updated_at': user.updated_at,
                'invited_by': user.invited_by,
                'last_login': user.last_login
            })
        
        return jsonify({'users': users_data})
        
    except Exception as e:
        logger.exception(f"Get users error: {e}")
        return jsonify({"error": f"Failed to get users: {str(e)}"}), 500

@app.route('/api/admin/users/<email>/status', methods=['PUT'])
@require_admin_token
def update_user_status(email):
    """Update user status (admin only)"""
    try:
        data = request.get_json()
        if not data or 'status' not in data:
            return jsonify({"error": "Status is required"}), 400
        
        try:
            new_status = UserStatus(data['status'])
        except ValueError:
            return jsonify({"error": "Invalid status"}), 400
        
        if user_manager.update_user_status(email, new_status):
            return jsonify({'message': f'User {email} status updated to {new_status.value}'})
        else:
            return jsonify({"error": "User not found"}), 404
            
    except Exception as e:
        logger.exception(f"Update user status error: {e}")
        return jsonify({"error": f"Failed to update user status: {str(e)}"}), 500

@app.route('/api/admin/users/<email>/role', methods=['PUT'])
@require_admin_token
def update_user_role(email):
    """Update user role (admin only)"""
    try:
        data = request.get_json()
        if not data or 'role' not in data:
            return jsonify({"error": "Role is required"}), 400
        
        try:
            new_role = UserRole(data['role'])
        except ValueError:
            return jsonify({"error": "Invalid role"}), 400
        
        if user_manager.update_user_role(email, new_role):
            return jsonify({'message': f'User {email} role updated to {new_role.value}'})
        else:
            return jsonify({"error": "User not found"}), 404
            
    except Exception as e:
        logger.exception(f"Update user role error: {e}")
        return jsonify({"error": f"Failed to update user role: {str(e)}"}), 500

@app.route('/api/admin/users/<email>', methods=['DELETE'])
@require_admin_token
def delete_user(email):
    """Delete a user (admin only)"""
    try:
        if user_manager.delete_user(email):
            return jsonify({'message': f'User {email} deleted successfully'})
        else:
            return jsonify({"error": "User not found"}), 404
            
    except Exception as e:
        logger.exception(f"Delete user error: {e}")
        return jsonify({"error": f"Failed to delete user: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

# Debug endpoint to check user status
@app.route('/api/debug/user/<email>')
def debug_user_status(email):
    """Debug endpoint to check user status"""
    try:
        user = user_manager.get_user(email)
        if user:
            return jsonify({
                "email": user.email,
                "status": user.status.value,
                "role": user.role.value,
                "created_at": user.created_at,
                "updated_at": user.updated_at,
                "last_login": user.last_login
            })
        else:
            return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Temporary endpoint to approve first admin user
@app.route('/api/approve-first-admin', methods=['POST'])
def approve_first_admin():
    """Temporary endpoint to approve the first admin user"""
    try:
        data = request.get_json()
        if not data or 'email' not in data:
            return jsonify({"error": "Email is required"}), 400
        
        email = data['email']
        user = user_manager.get_user(email)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Approve user and make them admin
        user_manager.update_user_status(email, UserStatus.APPROVED)
        user_manager.update_user_role(email, UserRole.ADMIN)
        
        return jsonify({
            "message": f"User {email} approved and promoted to admin",
            "user": {
                "email": email,
                "status": "approved",
                "role": "admin"
            }
        })
        
    except Exception as e:
        logger.exception(f"Approve first admin error: {e}")
        return jsonify({"error": f"Failed to approve user: {str(e)}"}), 500

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
