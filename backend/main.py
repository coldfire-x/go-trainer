from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import game

app = FastAPI(title="Go Trainer API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(game.router, prefix="/api", tags=["game"])

@app.get("/")
async def root():
    return {"message": "Welcome to Go Trainer API"}
