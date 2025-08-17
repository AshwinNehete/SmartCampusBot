from typing import List, Dict


class PromptTemplates:
    """Templates for generating prompts for the language model"""

    @staticmethod
    def create_qa_prompt(
        question: str, context_docs: List[Dict], max_context_length: int = 1500
    ) -> str:
        """
        Create a question-answering prompt with retrieved context

        Args:
            question: User's question
            context_docs: List of retrieved documents with answers
            max_context_length: Maximum length of context to include

        Returns:
            Formatted prompt
        """
        # Build context from retrieved documents
        context_parts = []
        current_length = 0

        for doc in context_docs:
            doc_text = f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}\n"
            if current_length + len(doc_text) <= max_context_length:
                context_parts.append(doc_text)
                current_length += len(doc_text)
            else:
                break

        context = "\n".join(context_parts)

        prompt = f"""You are SmartCampusBot, a helpful university assistant. Answer the student's question based on the provided context from the university FAQ database. 

Context from university FAQ:
{context}

Student Question: {question}

Instructions:
- Provide a clear, helpful answer based on the context provided
- If the context contains relevant information, use it to answer the question
- Keep your response concise but informative
- Maintain a friendly, professional tone appropriate for university students
- If you cannot find relevant information in the context, politely explain that you don't have that specific information

Answer:"""

        return prompt

    @staticmethod
    def create_fallback_prompt(question: str) -> str:
        """
        Create a fallback prompt when no relevant context is found

        Args:
            question: User's question

        Returns:
            Fallback prompt
        """
        prompt = f"""You are SmartCampusBot, a university assistant chatbot. A student has asked a question, but no relevant information was found in the university FAQ database.

Student Question: {question}

Instructions:
- Acknowledge that you don't have specific information about their question
- Suggest they contact the appropriate university department for help
- Provide general guidance on who they might contact (registrar, student services, academic advisor, etc.)
- Maintain a helpful and apologetic tone

Answer:"""

        return prompt

    @staticmethod
    def create_greeting_prompt() -> str:
        """Create a greeting message for the chatbot"""
        return """Hello! I'm SmartCampusBot, your university assistant. I'm here to help answer questions about:

📚 Class registration and course enrollment
💰 Tuition payments and financial aid
🏢 Campus facilities and services
🎓 Academic requirements and policies
🅿️ Parking and transportation
🏥 Health services and student support

What would you like to know about today?"""

    @staticmethod
    def create_category_prompt(question: str, category: str) -> str:
        """
        Create a prompt that includes category information

        Args:
            question: User's question
            category: Identified category of the question

        Returns:
            Category-aware prompt
        """
        category_context = {
            "registration": "class registration, course enrollment, add/drop procedures",
            "financial": "tuition payments, financial aid, scholarships, fees",
            "academic": "graduation requirements, transcripts, grades, academic policies",
            "facilities": "campus buildings, library, dining, recreational facilities",
            "services": "student services, health center, career services, ID cards",
            "technology": "WiFi, computer labs, IT support, online systems",
            "transportation": "parking, campus shuttle, transportation options",
        }

        context_desc = category_context.get(category, "university information")

        prompt = f"""You are SmartCampusBot answering a question related to {context_desc}.

Student Question: {question}

Provide a helpful response about university {context_desc}. If you need more specific information, suggest the student contact the relevant department.

Answer:"""

        return prompt
