from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import upload, chat, stats
from app.core.config import settings
from app.core.store import bot_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialise anything needed
    print(f"EzeeChatBot starting — model: {settings.GROQ_MODEL}")
    yield
    # Shutdown cleanup
    print("EzeeChatBot shutting down")


app = FastAPI(
    title="EzeeChatBot API",
    description="Upload PDFs, URLs or plain text and get a grounded chatbot instantly.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, tags=["Upload"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(stats.router, tags=["Stats"])


@app.get("/health")
async def health():
    return {"status": "ok", "bots_loaded": len(bot_store.list_bots())}
