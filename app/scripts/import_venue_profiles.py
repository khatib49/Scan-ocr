# scripts/import_venue_profiles.py
import os, json, asyncio
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "scan-invoice")
JSON_PATH = os.getenv("VENUE_PROFILES_PATH", "data/venue_profiles.json")

def _as_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return []

async def main():
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    coll = db["VenueProfile"]

    # Indexes
    await coll.create_index("MerchantId", unique=True, sparse=True)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert isinstance(data, list), "JSON must be an array"

    for p in data:
        # Normalize keyword fields to arrays
        p["MerchantName_Keyword"] = _as_list(p.get("MerchantName_Keyword"))
        p["MerchantAddress_Keyword"] = _as_list(p.get("MerchantAddress_Keyword"))

        mid = p.get("MerchantId")
        flt = {"MerchantId": mid} if mid is not None else {"_hash": hash(json.dumps(p, ensure_ascii=False))}

        await coll.update_one(flt, {"$set": p, "$setOnInsert": {"_seed": True}}, upsert=True)

    print("Import done.")

if __name__ == "__main__":
    asyncio.run(main())
