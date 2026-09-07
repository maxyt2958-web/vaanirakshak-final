"""
VaniRakshak — System Health & GPU Status Endpoint
"""

from __future__ import annotations

import torch
from fastapi import APIRouter

from backend.app.config import settings

router = APIRouter(tags=["System Health"])


@router.get("/health", summary="Service health & hardware state")
async def health_check():
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Host Execution)"
    vram_info = {}
    if cuda_available:
        vram_info = {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
            "target_budget_mb": 1200.0,
        }

    return {
        "status": "HEALTHY",
        "service": "VaniRakshak Forensic Audio Engine",
        "version": "1.0.0",
        "hardware": {
            "device": settings.DEVICE,
            "cuda_available": cuda_available,
            "device_name": device_name,
            "vram": vram_info,
        },
        "models": {
            "wavlm_model": settings.WAVLM_MODEL_ID,
            "wavlm_dim": settings.WAVLM_HIDDEN_DIM,
            "ecapa_model": settings.ECAPA_MODEL_ID,
            "ecapa_dim": settings.ECAPA_EMBED_DIM,
            "target_anchor_loaded": settings.ANCHOR_EMBEDDING_PATH.is_file(),
        },
        "dpdp_compliance": {
            "zero_raw_retention": True,
            "statute": "DPDP Act 2023 Section 8(7)",
        },
    }
