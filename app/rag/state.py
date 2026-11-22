from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route: Literal["kb", "web", "direct", "datetime"] = Field(
        description="""
        Choose the appropriate route:

        kb: Internal company IT support and company-specific policies.

        web: Latest news, current information, live prices,
        recent announcements, movie release dates and anything
        requiring up-to-date external information.

        direct: Greetings and general questions that do not
        require internal documents or current information.

        datetime:
        Use for questions asking:
        - Today's date
        - Current date
        - Current time
        - What day is today
        - What is the current day and time
        """
    )

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(description="Whether evidence is sufficient to answer")

class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: str
    trace: List[str]
    citations: List[dict]