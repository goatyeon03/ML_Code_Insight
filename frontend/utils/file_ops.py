import requests

API_URL = "http://localhost:8000"


def upload_code_api(user_id, project_id, cf, override_name=None):
    filename = override_name if override_name else cf.name

    files = {
        "file": (filename, cf.getvalue(), "text/x-python")
    }
    data = {"user_id": user_id, "project_id": project_id}

    try:
        res = requests.post(
            f"{API_URL}/upload_code",
            data=data,
            files=files,
            timeout=60,
        )
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def delete_file_api(user_id, file_id):
    """
    기존에는 frontend에서 직접 DB DELETE을 했음 (❌)
    이제는 백엔드 API를 호출해 삭제 (✔)
    """

    try:
        res = requests.post(
            f"{API_URL}/delete_file",
            data={"user_id": user_id, "file_id": file_id},
            timeout=15,
        )
        return res.json()
    except Exception as e:
        return {"error": str(e)}


def upload_result_api(user_id, project_id, rf):
    """
    Streamlit FileUploader 객체 rf를 FastAPI로 보냄.
    DB write는 오직 FastAPI만 수행.
    """

    files = {
        "file": (rf.name, rf.getvalue(), "application/json")
    }
    data = {"user_id": user_id, "project_id": project_id}

    try:
        res = requests.post(
            f"{API_URL}/upload_result",
            data=data,
            files=files,
            timeout=30,
        )
        return res.json()
    except Exception as e:
        return {"error": str(e)}

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
