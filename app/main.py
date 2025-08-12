from fastapi import FastAPI
from .routes.analyze import router as analyze_router
from .routes.prompts import router as prompts_router

app = FastAPI(title="Scan Invoice API")

@app.get("/")
async def root():
    return {"status": "ok"}

app.include_router(analyze_router)
app.include_router(prompts_router)