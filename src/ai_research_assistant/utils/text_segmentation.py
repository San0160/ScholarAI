# ai_research_assistant/utils/text_segmentation.py
from functools import lru_cache
from typing import Callable

import pysbd


@lru_cache(maxsize=4)
def _load_segmenter(language: str = "en"):
    return pysbd.Segmenter(language=language, clean=False)


def get_sentence_splitter(language: str = "en") -> Callable[[str], list[str]]:
    segmenter = _load_segmenter(language)
    return segmenter.segment