"""Online validation gateway across registries, search, domains, trademarks."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import quote

import diskcache
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from nomen.config import AppConfig, Secrets
from nomen.models import Candidate, Stage

_HIT = re.compile(
    r"\b(software|company|startup|repository|package|framework|cms|saas|platform|brand)\b",
    re.I,
)


class ValidationGateway:
    def __init__(self, cfg: AppConfig, secrets: Secrets, cache: diskcache.Cache) -> None:
        self.cfg = cfg
        self.secrets = secrets
        self.cache = cache
        self.sem = asyncio.Semaphore(max(4, cfg.concurrency))
        self.client = httpx.AsyncClient(
            timeout=cfg.timeout,
            follow_redirects=True,
            headers={"User-Agent": "NomenBrandEngine/4.0"},
            limits=httpx.Limits(
                max_connections=max(20, cfg.concurrency * 2),
                max_keepalive_connections=max(10, cfg.concurrency),
            ),
        )

    async def close(self) -> None:
        await self.client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.4, max=6),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
    )
    async def _get(
        self,
        url: str,
        *,
        cache_key: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        cached = self.cache.get(cache_key)
        if cached is not None:
            return _Cached(cached)
        async with self.sem:
            resp = await self.client.get(url, params=params, headers=headers)
        if resp.status_code in (429, 503):
            raise httpx.TransportError(f"rate-limited {resp.status_code}")
        if resp.status_code in (200, 404, 301, 302):
            try:
                payload = {
                    "status": resp.status_code,
                    "text": resp.text,
                    "json": resp.json(),
                }
            except Exception:
                payload = {"status": resp.status_code, "text": resp.text, "json": None}
            self.cache.set(cache_key, payload, expire=72 * 3600)
        return resp

    async def _exists(self, url: str, key: str) -> bool | None:
        try:
            r = await self._get(url, cache_key=key)
            if r.status_code == 404:
                return False
            if r.status_code in (200, 301, 302):
                return True
            return None
        except Exception:
            return None

    async def github(self, name: str) -> int | None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.secrets.github_token:
            headers["Authorization"] = f"Bearer {self.secrets.github_token}"
        r = await self._get(
            "https://api.github.com/search/repositories",
            cache_key=f"gh:{name}",
            params={"q": f'"{name}" in:name', "per_page": 1},
            headers=headers,
        )
        if r.status_code == 403:
            raise RuntimeError("GitHub rate limit; set GITHUB_TOKEN")
        if r.status_code != 200:
            return None
        return int((r.json() or {}).get("total_count", 0))

    async def search(self, query: str, kind: str) -> tuple[int | None, str | None]:
        if self.secrets.brave_search_api_key:
            r = await self._get(
                "https://api.search.brave.com/res/v1/web/search",
                cache_key=f"brave:{kind}:{query}",
                params={"q": query, "count": 8},
                headers={
                    "X-Subscription-Token": self.secrets.brave_search_api_key,
                    "Accept": "application/json",
                },
            )
            results = ((r.json() or {}).get("web") or {}).get("results") or []
            hits = 0
            for item in results:
                blob = f"{item.get('title', '')} {item.get('description', '')}"
                if _HIT.search(blob) or query.strip('"').lower() in blob.lower():
                    hits += 1
            return hits, "brave"

        if self.secrets.serpapi_key:
            r = await self._get(
                "https://serpapi.com/search.json",
                cache_key=f"serp:{kind}:{query}",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": self.secrets.serpapi_key,
                    "num": 8,
                },
            )
            data = r.json() or {}
            info = data.get("search_information") or {}
            if "total_results" in info:
                return int(info["total_results"]), "serpapi"
            return (1 if data.get("organic_results") else 0), "serpapi"

        if self.secrets.bing_search_api_key:
            r = await self._get(
                "https://api.bing.microsoft.com/v7.0/search",
                cache_key=f"bing:{kind}:{query}",
                params={"q": query, "count": 8},
                headers={"Ocp-Apim-Subscription-Key": self.secrets.bing_search_api_key},
            )
            webs = ((r.json() or {}).get("webPages") or {}).get("value") or []
            hits = sum(1 for w in webs if _HIT.search(f"{w.get('name','')} {w.get('snippet','')}"))
            return hits, "bing"

        # DuckDuckGo HTML-less instant answer (weak signal, no key)
        try:
            r = await self._get(
                "https://api.duckduckgo.com/",
                cache_key=f"ddg:{kind}:{query}",
                params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            )
            data = r.json() or {}
            text = f"{data.get('Heading','')} {data.get('Abstract','')} {data.get('AbstractText','')}"
            related = data.get("RelatedTopics") or []
            if text.strip() and _HIT.search(text):
                return 1, "duckduckgo"
            if related:
                return 1, "duckduckgo"
            return 0, "duckduckgo"
        except Exception:
            return None, None

    async def domain(self, host: str) -> bool | None:
        try:
            r = await self._get(
                f"https://rdap.org/domain/{quote(host)}",
                cache_key=f"rdap:{host}",
            )
            if r.status_code == 404:
                return False
            if r.status_code == 200:
                return True
            return None
        except Exception:
            return None

    async def opencorporates(self, name: str) -> int | None:
        if not self.secrets.opencorporates_api_token:
            return None
        r = await self._get(
            "https://api.opencorporates.com/v0.4/companies/search",
            cache_key=f"oc:{name}",
            params={"q": name, "api_token": self.secrets.opencorporates_api_token, "per_page": 5},
        )
        companies = ((r.json() or {}).get("results") or {}).get("companies") or []
        return sum(
            1
            for row in companies
            if name.lower() in ((row.get("company") or {}).get("name") or "").lower()
        )

    async def validate(self, candidate: Candidate) -> Candidate:
        name = candidate.name
        errors: list[str] = []

        async def cap(label: str, coro: Any) -> Any:
            try:
                return await coro
            except Exception as e:
                errors.append(f"{label}: {e}")
                return None

        checks = await asyncio.gather(
            cap("github", self.github(name)),
            cap("npm", self._exists(f"https://registry.npmjs.org/{quote(name)}", f"npm:{name}")),
            cap("pypi", self._exists(f"https://pypi.org/pypi/{quote(name)}/json", f"pypi:{name}")),
            cap("crates", self._exists(f"https://crates.io/api/v1/crates/{quote(name)}", f"crates:{name}")),
            cap("gem", self._exists(f"https://rubygems.org/api/v1/gems/{quote(name)}.json", f"gem:{name}")),
            cap(
                "nuget",
                self._exists(
                    f"https://api.nuget.org/v3-flatcontainer/{quote(name.lower())}/index.json",
                    f"nuget:{name}",
                ),
            ),
            cap(
                "packagist",
                self._get(
                    "https://packagist.org/search.json",
                    cache_key=f"pack:{name}",
                    params={"q": name},
                ),
            ),
            cap(
                "docker",
                self._get(
                    "https://hub.docker.com/v2/search/repositories/",
                    cache_key=f"docker:{name}",
                    params={"query": name, "page_size": 5},
                ),
            ),
            cap(
                "wordpress",
                self._get(
                    "https://api.wordpress.org/plugins/info/1.2/",
                    cache_key=f"wp:{name}",
                    params={"action": "plugin_information", "slug": name},
                ),
            ),
            cap(
                "gitlab",
                self._get(
                    "https://gitlab.com/api/v4/projects",
                    cache_key=f"gitlab:{name}",
                    params={"search": name, "per_page": 5},
                ),
            ),
        )

        (
            gh,
            npm,
            pypi,
            crates,
            gem,
            nuget,
            pack_resp,
            docker_resp,
            wp_resp,
            gitlab_resp,
        ) = checks

        packagist = False
        if pack_resp is not None and getattr(pack_resp, "status_code", None) == 200:
            results = (pack_resp.json() or {}).get("results") or []
            needle = name.lower().replace("-", "")
            packagist = any(needle in (x.get("name") or "").lower().replace("-", "") for x in results)

        docker = False
        if docker_resp is not None and getattr(docker_resp, "status_code", None) == 200:
            results = (docker_resp.json() or {}).get("results") or []
            docker = any(
                (x.get("repo_name") or x.get("name") or "").lower().split("/")[-1] == name.lower()
                for x in results
            )

        wp = False
        if wp_resp is not None and getattr(wp_resp, "status_code", None) == 200:
            data = wp_resp.json()
            wp = isinstance(data, dict) and bool(data.get("slug"))

        gitlab = False
        if gitlab_resp is not None and getattr(gitlab_resp, "status_code", None) == 200:
            rows = gitlab_resp.json() or []
            if isinstance(rows, list):
                gitlab = any(
                    (row.get("name") or "").lower() == name.lower()
                    or (row.get("path") or "").lower() == name.lower()
                    for row in rows
                )

        candidate.registries = {
            "github": gh,
            "npm": npm,
            "pypi": pypi,
            "crates": crates,
            "rubygems": gem,
            "nuget": nuget,
            "packagist": packagist,
            "dockerhub": docker,
            "wordpress": wp,
            "gitlab": gitlab,
        }

        # Domains
        domains: list[str] = []
        domain_states = await asyncio.gather(
            *[cap(f"domain.{t}", self.domain(f"{name}.{t}")) for t in self.cfg.tlds]
        )
        for tld, state in zip(self.cfg.tlds, domain_states):
            if state is True:
                domains.append(f"{name}.{tld}")
        candidate.domains_registered = domains

        # Search / company / trademark proxies
        web, company, software, tm = await asyncio.gather(
            cap("web", self.search(f'"{name}"', "web")),
            cap("company", self.search(f'"{name}" company OR startup OR software', "company")),
            cap("software", self.search(f'"{name}" software OR framework OR CMS OR SaaS', "software")),
            cap(
                "tm",
                self.search(
                    f'"{name}" trademark OR USPTO OR EUIPO OR "UK IPO" OR WIPO',
                    "tm",
                ),
            ),
        )
        oc = await cap("opencorporates", self.opencorporates(name))

        def hits(x: Any) -> int | None:
            if isinstance(x, tuple):
                return x[0]
            return None

        candidate.search_hits = {
            "web": hits(web),
            "company": hits(company),
            "software": hits(software),
            "trademark": hits(tm),
            "provider": (web[1] if isinstance(web, tuple) else None),
        }
        candidate.company_hits = (hits(company) or 0) + (oc or 0) if hits(company) is not None or oc is not None else None
        candidate.trademark_hits = hits(tm)
        candidate.errors = errors

        # Decision tree
        if (gh or 0) > 0:
            return candidate.reject(Stage.GITHUB, f"GitHub repositories={gh}")
        for label, val in [
            ("npm", npm),
            ("pypi", pypi),
            ("crates", crates),
            ("rubygems", gem),
            ("nuget", nuget),
            ("packagist", packagist),
            ("dockerhub", docker),
            ("wordpress", wp),
            ("gitlab", gitlab),
        ]:
            if val is True:
                return candidate.reject(Stage.REGISTRY, f"{label} exists")

        if self.cfg.strict and domains:
            return candidate.reject(Stage.DOMAIN, f"registered: {', '.join(domains)}")

        if (hits(web) or 0) > 0:
            return candidate.reject(Stage.SEARCH, f"web hits={hits(web)}")
        if (hits(software) or 0) > 0:
            return candidate.reject(Stage.SEARCH, f"software hits={hits(software)}")
        if (candidate.company_hits or 0) > 0:
            return candidate.reject(Stage.COMPANY, f"company hits={candidate.company_hits}")
        if (candidate.trademark_hits or 0) > 0:
            return candidate.reject(Stage.TRADEMARK, f"trademark hits={candidate.trademark_hits}")

        if self.cfg.strict and errors:
            return candidate.reject(Stage.REGISTRY, f"strict errors: {errors[0]}")

        return candidate.mark_clean()


class _Cached:
    def __init__(self, data: dict[str, Any]) -> None:
        self.status_code = int(data["status"])
        self.text = data.get("text") or ""
        self._json = data.get("json")

    def json(self) -> Any:
        return self._json
