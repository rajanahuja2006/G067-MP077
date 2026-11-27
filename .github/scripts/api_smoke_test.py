import json
import math
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE_URL = sys.argv[1].rstrip("/")


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()

    req = Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST"
    )

    try:
        with urlopen(req, timeout=10) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


# Test 1: Wait for the application to become healthy.
for attempt in range(30):
    try:
        status, body = request("/health")
        if status == 200 and body.get("status") == "ok":
            break
    except (URLError, TimeoutError):
        pass

    time.sleep(2)
else:
    raise RuntimeError("Application health check failed")

print("PASS: Health endpoint")


# Test 2: Valid prediction.
payload = {
    "flu_cases_reported": 25,
    "temp_c": 24.5,
    "nearby_clinics": 5,
    "discount_pct": 10,
    "stock_lines": 150,
    "distance_to_hospital_km": 2.5,
    "shop_type": "chain",
    "locality": "Old Town"
}

status, body = request("/predict", payload)

assert status == 200, (status, body)
assert isinstance(body.get("prediction"), (int, float))
assert math.isfinite(body["prediction"])

print("PASS: Valid prediction:", body["prediction"])


# Test 3: Invalid clinic count.
invalid_clinics = {**payload, "nearby_clinics": 26}

status, body = request("/predict", invalid_clinics)

assert status == 422, (status, body)

print("PASS: Invalid clinic count rejected")


# Test 4: Invalid discount.
invalid_discount = {**payload, "discount_pct": -5}

status, body = request("/predict", invalid_discount)

assert status == 422, (status, body)

print("PASS: Invalid discount rejected")

print("All four API smoke tests passed successfully.")
