"""Small stdlib HTTP/SSE helpers shared by the chat and native SGLang clients."""

import json
import time
import urllib.error
import urllib.request


def opener(api_key=""):
    # Cluster requests should not be silently redirected through a user proxy.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if api_key:
        client.addheaders = [("Authorization", f"Bearer {api_key}")]
    return client


def get_json(client, url, timeout):
    with client.open(url, timeout=timeout) as response:
        return json.load(response)


def post_request(url, body):
    return urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )


def events(response):
    """Yield (payload, monotonic_time); None means SSE [DONE].

    SSE fields may span lines. An output event is not assumed to be one token.
    """
    data = []
    for raw in response:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if not data:
                continue
            payload = "\n".join(data)
            data.clear()
            if payload == "[DONE]":
                yield None, time.perf_counter()
                return
            yield json.loads(payload), time.perf_counter()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        payload = "\n".join(data)
        yield (
            None if payload == "[DONE]" else json.loads(payload)
        ), time.perf_counter()


def error_text(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:1000]}"
    return f"{type(exc).__name__}: {exc}"
