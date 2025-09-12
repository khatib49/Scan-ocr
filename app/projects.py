# app/projects.py
from datetime import datetime, timezone
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, HTTPException, Request, Query, Security
from bson import ObjectId
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime

from app.security import verify_admin_key, _mongo_db as DB
from utils.helpers import (
    ensure_project_indexes,
    generate_unique_api_key,
    normalize_project_out,
    parse_oid,
)

router = APIRouter(prefix="/projects", tags=["Projects"])

# ----- Schemas -----
class ProjectBase(BaseModel):
    Name: str = Field(min_length=1, max_length=120)

class ProjectCreate(ProjectBase): ...
class ProjectUpdate(BaseModel):
    Name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    rotateApiKey: Optional[bool] = False

class ProjectOut(ProjectBase):
    model_config = ConfigDict(populate_by_name=True)
    id: str = Field(alias="_id")
    ApiKey: str
    CreatedAt: datetime | str

    @field_validator("CreatedAt", mode="before")
    @classmethod
    def _coerce_created_at(cls, v):
        if isinstance(v, datetime) or v is None:
            return v
        if isinstance(v, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ",
                        "%Y-%m-%dT%H:%M:%S%z",
                        "%m/%d/%Y %I:%M:%S %p",
                        "%m/%d/%Y %H:%M:%S"):
                try:
                    return datetime.strptime(v, fmt)
                except Exception:
                    pass
        return v

# ----- Self-service (scoped to caller's project) -----

@router.get("/me", response_model=ProjectOut, summary="Get my project")
async def get_my_project(request: Request) -> Any:
    proj_id: ObjectId = request.state.project["_id"]
    doc = await DB["Project"].find_one({"_id": proj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectOut(**normalize_project_out(doc))

@router.patch("/me", response_model=ProjectOut, summary="Update my project (rename and/or rotate key)")
async def update_my_project(request: Request, patch: ProjectUpdate) -> Any:
    pid: ObjectId = request.state.project["_id"]
    update: Dict[str, Any] = {}
    if patch.Name is not None:
        update["Name"] = patch.Name.strip()
    if patch.rotateApiKey:
        update["ApiKey"] = await generate_unique_api_key(DB)

    if not update:
        return ProjectOut(**normalize_project_out(request.state.project))

    result = await DB["Project"].find_one_and_update(
        {"_id": pid},
        {"$set": update},
        return_document=True  # Motor accepts ReturnDocument or truthy for "after"
    )
    if not result:
        raise HTTPException(status_code=404, detail="Project not found")

    request.state.project = result
    return ProjectOut(**normalize_project_out(result))

@router.delete("/me", status_code=204, summary="Delete my project")
async def delete_my_project(request: Request) -> None:
    pid: ObjectId = request.state.project["_id"]
    await DB["Project"].delete_one({"_id": pid})

# ----- Admin CRUD for all projects -----
# NOTE: use Security(...) so Swagger exposes X-Admin-Key and sends it.

@router.get(
    "",
    response_model=List[ProjectOut],
    dependencies=[Security(verify_admin_key)],
    summary="List projects (admin)",
)
async def list_projects(
    q: Optional[str] = Query(None, description="Case-insensitive substring match on Name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Any:
    flt: Dict[str, Any] = {}
    if q:
        flt["Name"] = {"$regex": q, "$options": "i"}

    cursor = DB["Project"].find(flt).sort("CreatedAt", -1).skip(offset).limit(limit)
    docs = [normalize_project_out(d) async for d in cursor]
    return [ProjectOut(**d) for d in docs]

@router.get(
    "/{project_id}",
    response_model=ProjectOut,
    dependencies=[Security(verify_admin_key)],
    summary="Get project (admin)",
)
async def get_project(project_id: str) -> Any:
    doc = await DB["Project"].find_one({"_id": parse_oid(project_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectOut(**normalize_project_out(doc))

@router.post(
    "",
    response_model=ProjectOut,
    status_code=201,
    dependencies=[Security(verify_admin_key)],
    summary="Create project (admin)",
)
async def create_project(payload: ProjectCreate) -> Any:
    await ensure_project_indexes(DB)
    api_key = await generate_unique_api_key(DB)
    now = datetime.now(timezone.utc)
    doc = {"Name": payload.Name.strip(), "ApiKey": api_key, "CreatedAt": now}
    res = await DB["Project"].insert_one(doc)
    created = await DB["Project"].find_one({"_id": res.inserted_id})
    return ProjectOut(**normalize_project_out(created))

@router.patch(
    "/{project_id}",
    response_model=ProjectOut,
    dependencies=[Security(verify_admin_key)],
    summary="Update project (admin)",
)
async def update_project(project_id: str, patch: ProjectUpdate) -> Any:
    oid = parse_oid(project_id)
    update: Dict[str, Any] = {}
    if patch.Name is not None:
        update["Name"] = patch.Name.strip()
    if patch.rotateApiKey:
        update["ApiKey"] = await generate_unique_api_key(DB)

    if not update:
        doc = await DB["Project"].find_one({"_id": oid})
        if not doc:
            raise HTTPException(status_code=404, detail="Project not found")
        return ProjectOut(**normalize_project_out(doc))

    doc = await DB["Project"].find_one_and_update(
        {"_id": oid}, {"$set": update}, return_document=True
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return ProjectOut(**normalize_project_out(doc))

@router.delete(
    "/{project_id}",
    status_code=204,
    dependencies=[Security(verify_admin_key)],
    summary="Delete project (admin)",
)
async def delete_project(project_id: str) -> None:
    await DB["Project"].delete_one({"_id": parse_oid(project_id)})
