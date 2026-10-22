
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.analysis import router as analysis_router

app = FastAPI(
    title="TestWeaver API",
    description="Backend API for intelligent test impact analysis.",
    version="0.1.0",
)

# Allow the local React/Vite frontend to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "success",
        "message": "TestWeaver backend is running",
    }


app.include_router(analysis_router)
