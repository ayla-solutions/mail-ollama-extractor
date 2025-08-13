import os, requests, json
from typing import Dict, Any
from msal import ConfidentialClientApplication

# Prefer the DATAVERSE_* names you provided, fall back to legacy names if present
DV_TENANT_ID    = os.getenv("DATAVERSE_TENANT_ID")    or os.getenv("TENANT_ID")
DV_CLIENT_ID    = os.getenv("DATAVERSE_CLIENT_ID")    or os.getenv("CLIENT_ID")
DV_CLIENT_SECRET= os.getenv("DATAVERSE_CLIENT_SECRET")or os.getenv("CLIENT_SECRET")
DV_RESOURCE     = (os.getenv("DATAVERSE_RESOURCE")    or os.getenv("DATAVERSE_URL") or "").rstrip("/")
TABLE           = os.getenv("DATAVERSE_TABLE")  # e.g., crabb_arth_main1s

def col(name: str) -> str:
    # Column logical name resolver; values come from the environment (e.g., COL_CATEGORY, etc.)
    return os.getenv(name)

AUTHORITY = f"https://login.microsoftonline.com/{DV_TENANT_ID}"
SCOPE     = f"{DV_RESOURCE}/.default"

_app = ConfidentialClientApplication(
    client_id=DV_CLIENT_ID,
    authority=AUTHORITY,
    client_credential=DV_CLIENT_SECRET,
)

def _token() -> str:
    r = _app.acquire_token_silent([SCOPE], account=None)
    if not r:
        r = _app.acquire_token_for_client(scopes=[SCOPE])
    if "access_token" not in r:
        raise RuntimeError(f"Dataverse auth failed: {r.get('error_description') or r}")
    return r["access_token"]

def create_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not (DV_RESOURCE and TABLE):
        raise RuntimeError("DATAVERSE_RESOURCE or DATAVERSE_TABLE not set")
    url = f"{DV_RESOURCE}/api/data/v9.2/{TABLE}"
    tok = _token()
    h = {
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json; charset=utf-8",
        "OData-MaxVersion": "4.0",
        "OData-Version": "4.0",
        "Prefer": "return=representation",
    }
    resp = requests.post(url, headers=h, data=json.dumps(payload))
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Dataverse create failed: {resp.status_code} {resp.text}")
    return {"ok": True, "status": resp.status_code}
