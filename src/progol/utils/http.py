import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session(
    total_retries: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    timeout: int = 30,
) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("GET", "HEAD", "OPTIONS"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.request_timeout = timeout
    return s


def api_football_session() -> requests.Session:
    s = make_session()
    s.headers.update({"x-apisports-key": os.getenv("FOOTBALL_API_KEY", "")})
    return s
