# helpers/projects.py
from fastapi import HTTPException
from bson import ObjectId

async def get_project_by_api_key(db, api_key: str):
    project = await db["Project"].find_one({"ApiKey": api_key})
    if not project:
        # API key is valid format (already passed security) but not mapped to any Project
        raise HTTPException(status_code=403, detail="API key not assigned to a project")
    return project  # contains _id: ObjectId(...)
