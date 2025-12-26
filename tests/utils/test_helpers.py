import os
import json
import pytest
from src.utils.helpers import (
    clean_text, 
    extract_keywords, 
    categorize_query, 
    format_response_time,
    load_json_file,
    save_json_file,
    truncate_text
)

class TestHelpers:
    def test_clean_text(self):
        assert clean_text("") == ""
        assert clean_text("  hello   world  ") == "hello world"
        assert clean_text("hello! @# world") == "hello!  world" # helpers.py removes chars but doesn't re-collapse spaces

    def test_extract_keywords(self):
        text = "The quick brown fox jumps over the lazy dog"
        keywords = extract_keywords(text)
        # "the" is a stop word
        assert "the" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords

    def test_categorize_query(self):
        assert categorize_query("How do I register for classes?") == "registration"
        assert categorize_query("Where is the library?") == "facilities"
        assert categorize_query("What is the tuition cost?") == "financial"
        assert categorize_query("My wifi is not working") == "technology"
        assert categorize_query("Unknown query about space") == "general"

    def test_format_response_time(self):
        assert format_response_time(0.5) == "500ms"
        assert format_response_time(1.5) == "1.5s"
        assert format_response_time(65.5) == "1m 5.5s"

    def test_truncate_text(self):
        assert truncate_text("Short text", 20) == "Short text"
        assert truncate_text("Long text that needs truncation", 10, "...") == "Long te..."

    def test_json_file_ops(self, tmp_path):
        data = {"key": "value"}
        file_path = tmp_path / "test.json"
        
        # Test save
        assert save_json_file(data, str(file_path)) is True
        assert file_path.exists()
        
        # Test load
        loaded_data = load_json_file(str(file_path))
        assert loaded_data == data

    def test_load_json_file_not_found(self):
        assert load_json_file("non_existent_file.json") is None

    def test_load_json_file_invalid(self, tmp_path):
        file_path = tmp_path / "invalid.json"
        with open(file_path, "w") as f:
            f.write("{invalid json")
        
        assert load_json_file(str(file_path)) is None
