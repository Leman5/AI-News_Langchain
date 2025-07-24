"""News fetching module using Event Registry / NewsAPI.ai.

This module defines a simple wrapper class around the Event Registry
(`newsapi.ai`) REST API.  The API endpoint we use is the
``/api/v1/article/getArticles`` endpoint, which returns a page of
news articles matching a set of filters and search conditions.  The
official documentation explains that you can filter on many fields
such as keywords, concept URIs, category URIs, source URIs and
dates【214396105881242†screenshot】.  Each API call can return up to
100 articles at a time, and pagination is handled via the
``articlesPage`` parameter【214396105881242†screenshot】.  In this
application we query recent articles from Australian news outlets by
setting the ``sourceLocationUri`` to the Wikipedia page for
Australia and restricting the language to English.  The API
response includes a JSON object with a list of article results and
associated metadata.

Note:  This module does not implement any retry logic or rate
limiting.  If you intend to deploy this in production you should
augment the ``fetch_articles`` method with appropriate error
handling and back‑off to respect API quotas.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional

import requests


class NewsFetcher:
    """Simple client for fetching articles from NewsAPI.ai via Event Registry."""

    BASE_URL = "https://eventregistry.org/api/v1/article/getArticles"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_articles(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        page: int = 1,
        count: int = 100,
        source_location_uri: Optional[str] = None,
        language: str = "eng",
        duplicate_filter: str = "skipDuplicates",
        include_pr_and_blogs: bool = False,
    ) -> List[Dict[str, any]]:
        """Fetch a page of articles published between ``start_date`` and ``end_date``.

        The Event Registry API accepts a large number of optional parameters to
        control which articles are returned.  For our use case we restrict
        results to the specified time window, English language and (optionally)
        a particular source location.  See the API documentation for
        details on the available filters【436125224526444†screenshot】.  Duplicate
        articles can be suppressed by setting ``duplicate_filter`` to
        ``"skipDuplicates"``【571414561824264†screenshot】.

        Parameters
        ----------
        start_date, end_date: datetime.date
            The inclusive date range over which to search for articles.  The
            API expects dates in ``YYYY-MM-DD`` format【525397756974300†screenshot】.
        page: int, optional
            Results page number (starting from 1)【214396105881242†screenshot】.
        count: int, optional
            Number of articles to return (maximum 100 per request)【214396105881242†screenshot】.
        source_location_uri: str, optional
            Wikipedia URI for a geographic location from which articles should
            originate.  For example, use ``"http://en.wikipedia.org/wiki/Australia"``
            to restrict results to Australian sources【125145694911606†screenshot】.
        language: str, optional
            ISO3 language code to restrict articles by language (default 'eng')【525397756974300†screenshot】.
        duplicate_filter: str, optional
            Behaviour when encountering duplicates.  Supported values include
            ``"skipDuplicates"``, ``"keepOnlyDuplicates"`` and ``"keepAll"``【571414561824264†screenshot】.
        include_pr_and_blogs: bool, optional
            Whether to include press releases and blog posts in addition to
            news articles.  When ``True`` the ``dataType`` parameter will
            include ``pr`` and ``blog`` as allowed values【597563020955336†screenshot】.

        Returns
        -------
        List[Dict[str, any]]
            A list of article objects as returned by the API.  Each article
            contains keys such as ``title``, ``body``, ``source`` and
            ``date``.
        """
        params: Dict[str, any] = {
            "apiKey": self.api_key,
            "resultType": "articles",
            "articlesPage": page,
            "articlesCount": count,
            "lang": language,
            "dateStart": start_date.strftime("%Y-%m-%d"),
            "dateEnd": end_date.strftime("%Y-%m-%d"),
            "articleBodyLen": -1,  # return full article bodies【597563020955336†screenshot】
            "isDuplicateFilter": duplicate_filter,
        }
        # restrict content types
        if include_pr_and_blogs:
            params["dataType"] = ["news", "pr", "blog"]
        else:
            params["dataType"] = ["news"]
        # restrict sources by location
        if source_location_uri:
            params["sourceLocationUri"] = source_location_uri

        response = requests.get(self.BASE_URL, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch articles: {response.status_code} {response.text[:200]}"
            )
        data = response.json()
        # the returned JSON structure contains articles under
        # data['articles']['results'] when resultType='articles'
        articles = data.get("articles", {}).get("results", [])
        return articles