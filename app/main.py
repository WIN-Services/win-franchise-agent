from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.router import router
from app.config import settings

app = FastAPI(
    title="Franchise Chatbot System API",
    description="Backend API for Strategic Partner acquisition and onboarding.",
    version="1.0.0"
)

# Allow Expo web dev server + any local client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router for our chat endpoints
app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.LLM_MODEL}
