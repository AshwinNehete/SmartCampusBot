import pytest
from src.generator.prompt_templates import PromptTemplates

class TestPromptTemplates:
    def test_create_qa_prompt(self):
        question = "Test Question"
        context_docs = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"}
        ]
        
        prompt = PromptTemplates.create_qa_prompt(question, context_docs)
        
        assert "Test Question" in prompt
        assert "Q1" in prompt
        assert "A1" in prompt
        assert "Q2" in prompt
        assert "A2" in prompt

    def test_create_fallback_prompt(self):
        prompt = PromptTemplates.create_fallback_prompt("Help me")
        assert "Help me" in prompt
        assert "no relevant information was found" in prompt

    def test_create_greeting_prompt(self):
        prompt = PromptTemplates.create_greeting_prompt()
        assert "SmartCampusBot" in prompt
        assert "Class registration" in prompt

    def test_create_category_prompt(self):
        prompt = PromptTemplates.create_category_prompt("Test Q", "registration")
        assert "Test Q" in prompt
        assert "class registration" in prompt
