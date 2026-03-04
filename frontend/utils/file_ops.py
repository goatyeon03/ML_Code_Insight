import os
import requests

API_URL = os.genenv("API_URL", "http://localhost:8000").rstrip("/")


def upload_code_api(user_id, project_id, cf, override_name=None):
    filename = override_name if override_name else cf.name

    files = {
        "file": (filename, cf.getvalue(), "text/x-python")
    }
    data = {
        "user_id": user_id, 
        "project_id": project_id,
        "override_name": filename,
    }

    r= requests.post(f"{API_URL}/uploda_code", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()



def upload_result_api(user_id, project_id, rf, override_name=None):
    """
    Streamlit FileUploader 객체 rf를 FastAPI로 보냄.
    DB write는 오직 FastAPI만 수행.
    """
    filename = override_name if override_name else rf.name

    files = {
        "file": (filename, rf.getvalue(), "application/json")
    }
    data = {
        "user_id": user_id, 
        "project_id": project_id,
        "override_name": filename,    
    }

    r = requests.post(f"{API_URL}/upload_result", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def create_project_api(user_id, project_name):
    try:
        res = requests.post(
            f"{API_URL}/create_project",
            data={"user_id": user_id, "project_name": project_name}
        )
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def delete_project_api(user_id, project_id):
    try:
        resp = requests.post(
            f"{API_URL}/delete_project",
            data={"user_id": user_id, "project_id": project_id},
            timeout=10
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}
    
def delete_account_api(user_id: int):
    import requests
    url = f"{API_URL}/delete_account"
    res = requests.delete(url, json={"user_id": user_id})
    try:
        return res.json()
    except:
        return {"error": "Invalid response from server"}
