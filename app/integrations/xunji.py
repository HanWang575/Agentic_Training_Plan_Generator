"""Small read-only client for the Xunji training Open API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib import error, request
import gzip
import json
import re

from app.core.config import Settings
from app.core.errors import ConfigurationError


SCHEMA_VERSION = "train_open_api_v2"
READ_ENDPOINT = "/api_trains_for_llm_v2"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(slots=True)
class XunjiClient:
    """Read training logs from Xunji with local date-based caching."""

    settings: Settings

    def read_day(
        self,
        datestr: str,
        include_full_data: bool = True,
        force_refresh: bool = False,
    ) -> dict:
        """Read one Xunji training date, using the local cache unless forced."""

        if not DATE_RE.match(datestr):
            raise ValueError("datestr must use YYYY-MM-DD.")
        cache_path = self._cache_path(datestr, include_full_data)
        if cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        api_key = self.settings.xunji_api_key
        if not api_key:
            raise ConfigurationError("XUNJI_API_KEY is not configured.")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "datestr": datestr,
            "include_full_data": bool(include_full_data),
        }
        result = self._post_json(READ_ENDPOINT, payload, api_key)
        if result.get("success") is True or "res" in result:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    def _cache_path(self, datestr: str, include_full_data: bool) -> Path:
        suffix = "full" if include_full_data else "light"
        return self.settings.xunji_cache_dir / f"{datestr}_{suffix}.json"

    def _post_json(self, endpoint: str, payload: dict, api_key: str) -> dict:
        req = request.Request(
            self.settings.xunji_base_url.rstrip("/") + endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=40) as response:
                body_bytes = response.read()
                encoding = response.headers.get("Content-Encoding", "").lower()
        except error.HTTPError as exc:
            body_bytes = exc.read()
            if body_bytes.startswith(b"\x1f\x8b"):
                try:
                    body_bytes = gzip.decompress(body_bytes)
                except OSError:
                    pass
            body = body_bytes.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as parse_exc:
                raise RuntimeError(f"Xunji API HTTP {exc.code}: {body}") from parse_exc
            raise RuntimeError(f"Xunji API HTTP {exc.code}: {json.dumps(parsed, ensure_ascii=False)}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Xunji API network error: {exc.reason}") from exc

        if encoding == "gzip" or body_bytes.startswith(b"\x1f\x8b"):
            body_bytes = gzip.decompress(body_bytes)
        body = body_bytes.decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Xunji API returned non-JSON response: {body}") from exc
