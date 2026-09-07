"""
VaniRakshak — API Router Aggregation
"""

from fastapi import APIRouter

from backend.app.api.endpoints.analyze import router as analyze_router
from backend.app.api.endpoints.enroll import router as enroll_router
from backend.app.api.endpoints.health import router as health_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(enroll_router)
api_router.include_router(analyze_router)
