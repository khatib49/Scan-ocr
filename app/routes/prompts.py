from fastapi import APIRouter
from pydantic import BaseModel
from ..prompt_store import get_active_prompt, set_active_prompt

router = APIRouter(prefix="/prompts", tags=["prompts"])

class SetPrompt(BaseModel):
    system_prompt: str

@router.get("/active")
async def get_prompt():
    return {"system_prompt": await get_active_prompt()}

@router.post("/active")
async def set_prompt(req: SetPrompt):
    ver = await set_active_prompt(req.system_prompt)
    return {"message": "Prompt updated", "version": ver}
