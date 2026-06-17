import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock


class _FakeNltkData:
    def __init__(self, available_resources):
        self.available_resources = set(available_resources)
        self.checked_resources = []

    def find(self, resource):
        self.checked_resources.append(resource)
        if resource in self.available_resources:
            return resource
        raise LookupError(resource)


def _fake_nltk(available_resources):
    return SimpleNamespace(
        data=_FakeNltkData(available_resources),
        download=Mock(return_value=False),
    )


def _synset_utils():
    return importlib.import_module("molmo_spaces.utils.synset_utils")


def test_nltk_corpus_available_accepts_extracted_corpus():
    nltk = _fake_nltk({"corpora/wordnet"})

    assert _synset_utils()._nltk_corpus_available(nltk, "wordnet")
    assert nltk.data.checked_resources == ["corpora/wordnet"]


def test_nltk_corpus_available_accepts_zip_corpus():
    nltk = _fake_nltk({"corpora/wordnet.zip"})

    assert _synset_utils()._nltk_corpus_available(nltk, "wordnet")
    assert nltk.data.checked_resources == [
        "corpora/wordnet",
        "corpora/wordnet.zip",
    ]


def test_ensure_nltk_skips_download_for_available_corpora(monkeypatch):
    nltk = _fake_nltk({"corpora/wordnet.zip", "corpora/wordnet2022"})
    monkeypatch.setitem(sys.modules, "nltk", nltk)

    _synset_utils()._ensure_nltk()

    nltk.download.assert_not_called()


def test_ensure_nltk_downloads_only_missing_corpus(monkeypatch):
    nltk = _fake_nltk({"corpora/wordnet2022"})
    monkeypatch.setitem(sys.modules, "nltk", nltk)

    _synset_utils()._ensure_nltk()

    nltk.download.assert_called_once_with(
        "wordnet",
        quiet=True,
        raise_on_error=False,
    )


def test_synset_utils_import_uses_local_corpora_without_download(monkeypatch):
    import nltk

    download = Mock(side_effect=AssertionError("unexpected nltk.download call"))
    monkeypatch.setattr(nltk, "download", download)
    sys.modules.pop("molmo_spaces.utils.synset_utils", None)

    module = importlib.import_module("molmo_spaces.utils.synset_utils")

    download.assert_not_called()
    assert module.wn.synset("physical_entity.n.01").name() == "physical_entity.n.01"
