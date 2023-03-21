import requests
import urllib.parse
from typing import Any
from duckduckgo_search import ddg, ddg_answers, ddg_images, ddg_videos, ddg_news, ddg_maps, ddg_translate, ddg_suggestions


class WebSearch():
    """
    Class to handle web search
    """
    
    def __init__(self) -> None:
        """
        Initialize web search
        """
        self.cache = {}

    def search(self, keywords: Any, region: str = "us-en", safesearch: str = "Off", time: Any | None = 'y', max_results: Any | None = 20, page: int = 1, output: Any | None = None, download: bool = False, cache: bool = False) -> list:
        """
        Search

        Parameters:
            keywords (Any): keywords
            region (str): region
            safesearch (str): safesearch
            time (Any | None): time (one of: 'd', 'w', 'm', 'y')
            max_results (Any | None): max results
            page (int): page
            output (Any | None): output
            download (bool): download
            cache (bool): If True, cache results

        Returns:
            list: results
        """
        if cache and 'search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download) in self.cache:
            return self.cache['search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download)]
        response = ddg(keywords=keywords, region=region, safesearch=safesearch, time=time, max_results=max_results, page=page, output=output, download=download)
        if cache:
            self.cache['search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download)] = response
        return response


class WolframAlpha():
    """
    Class to handle wolfram alpha api requests
    """
    def __init__(self, app_id: str) -> None:
        """
        Initialize wolfram alpha

        Parameters:
            app_id (str): app id
        """
        self.app_id = app_id
        self.cache = {}
        self.endpoints = {
            'short_answer': 'https://api.wolframalpha.com/v1/result?i=',
        }

    def get_short_answer(self, query: str, cache: bool = False) -> str:
        """
        Get short answer result

        Parameters:
            query (str): query
            cache (bool): If True, cache results

        Returns:
            str: result
        """
        if cache and 'get_short_answer' + str(query) in self.cache:
            return self.cache['get_short_answer' + str(query)]
        response = requests.get(self.endpoints['short_answer'] + urllib.parse.quote(query) + '&appid=' + self.app_id)
        response = {
            'query': query,
            'response': response.text,
        }
        if cache:
            self.cache['get_short_answer' + str(query)] = response
        return response
