import threading
from typing import List, Optional

_tls = threading.local()

def set_current_req_ids(req_ids: List[str]):
    """
    req_ids: length == num_tokens in this forward
    """
    _tls.req_ids = req_ids

def get_current_req_ids() -> Optional[List[str]]:
    return getattr(_tls, "req_ids", None)

def clear_current_req_ids():
    if hasattr(_tls, "req_ids"):
        del _tls.req_ids
