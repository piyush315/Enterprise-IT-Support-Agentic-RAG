import logging
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None


def llm():
    global _llm

    if _llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")

        _llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    return _llm


def web_search_tool():
    global _web_search

    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


# ---------------------------------------------------------
# 1. ROUTER
# ---------------------------------------------------------


def route_question(state: AgentState):
    router = llm().with_structured_output(
        RouteDecision,
        method="json_mode",
    )

    decision = router.invoke(f"""
You are an intelligent router for an enterprise IT support assistant.

Choose exactly one route:

1. kb:
Use for questions about information specific to the user's company:
- Company IT policies
- VPN and remote access
- Password reset and MFA
- Laptop setup
- Software access
- Internal security policies
- Company email and devices
- Internal IT troubleshooting

2. web:
Use for questions requiring current, recent, or externally verifiable
information, including:
- Latest AI or technology news
- Current software/library versions
- Recent product or company announcements
- Current prices, exchange rates, or market information
- Movie release dates and current entertainment news
- Current government policies, laws, or public announcements
- Events happening today or this week
- Any fact that may have changed since your training data

3. datetime:
Use ONLY for questions asking about the current system date or time:
- Today's date
- Current date
- What day is today?
- Current time
- Current date and time

Examples:
"Give me today's date" → datetime
"What is the current time?" → datetime
"What is the latest AI news?" → web
"What is our current company policy?" → kb

4. direct:
Use only for greetings, thanks, casual conversation, and stable
general-knowledge questions that do not require company documents
or current external information.

Important routing rules:
- Company-specific questions must use kb.
- Current external information must use web.
- Current system date/time questions must use datetime.
- Do not use direct for questions requiring live information.
- If uncertain whether external information may have changed, prefer web.

Question: {state["question"]}

Return valid JSON with exactly this field:
{{"route": "kb"}}
The route value must be kb, web, direct, or datetime.
""")

    route = decision.route

    logger.info("Question routed to: %s", route)

    return {
        "source_used": route,
        "trace": add_trace(state, f"Router → {route.upper()}"),
    }



# IMPORTANT:
# This function must explicitly handle all three routes.
def route_after_router(
    state: AgentState,
) -> Literal["retrieve_kb", "search_web", "direct_answer", "get_current_datetime"]:

    route = state["source_used"]

    if route == "kb":
        return "retrieve_kb"

    if route == "web":
        return "search_web"

    if route == "datetime":
        return "get_current_datetime"

    return "direct_answer"


# ---------------------------------------------------------
# 2. PRIVATE KNOWLEDGE BASE RETRIEVAL
# ---------------------------------------------------------

def retrieve_kb(state: AgentState):
    query = state.get("current_query") or state["question"]

    docs = get_retriever().invoke(query)

    logger.info("Private KB returned %s chunks", len(docs))

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks",
        ),
    }


def grade_kb(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in state.get("kb_docs", [])
    )

    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.

Question:
{state["question"]}

Private company knowledge-base evidence:
{context}

Return good only if the evidence is sufficient, relevant, and specific
enough to answer the question confidently.

Otherwise return weak.

Return valid JSON:
{{"grade": "good"}}
or
{{"grade": "weak"}}
""")

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.grade.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:

    return (
        "generate_from_kb"
        if state["kb_grade"] == "good"
        else "search_web"
    )


# ---------------------------------------------------------
# 3. TAVILY WEB SEARCH
# ---------------------------------------------------------

def search_web(state: AgentState):
    query = state.get("current_query") or state["question"]

    logger.info("Starting Tavily search: %s", query)

    result = web_search_tool().invoke({"query": query})

    lines = []
    citations = []

    if isinstance(result, dict):

        search_answer = result.get("answer")

        if search_answer:
            lines.append(f"Search answer:\n{search_answer}")

        for item in result.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            published_date = (
                item.get("published_date")
                or item.get("date")
                or ""
            )

            lines.append(
                f"Title: {title}\n"
                f"Published date: {published_date}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            if url:
                citations.append({
                    "title": title or url,
                    "url": url,
                    "type": "web",
                    "published_date": published_date,
                })

    elif isinstance(result, list):

        for item in result:
            if isinstance(item, dict):
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("content", "")

                lines.append(
                    f"Title: {title}\nURL: {url}\nContent: {content}"
                )

                if url:
                    citations.append({
                        "title": title or url,
                        "url": url,
                        "type": "web",
                    })
            else:
                lines.append(str(item))

    else:
        lines.append(str(result))

    web_results = "\n\n".join(lines)

    if not web_results.strip():
        web_results = "Tavily did not return usable search results."

    logger.info(
        "Tavily search completed. Results: %s, citations: %s",
        len(lines),
        len(citations),
    )

    return {
        "web_results": web_results,
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            f"Tavily web search completed → {len(citations)} sources",
        ),
    }

def get_current_datetime(state: AgentState):

    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    answer = now.strftime(
        "Today's date is %d %B %Y.\n"
        "Current time is %I:%M %p IST."
    )

    return {
        "answer": answer,
        "source_used": "system",
        "citations": [],
        "trace": add_trace(
            state,
            "Current date/time → Python system clock"
        )
    }

# ---------------------------------------------------------
# 4. WEB EVIDENCE GRADING
# ---------------------------------------------------------

def grade_web(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    grade = grader.invoke(f"""
You evaluate external web-search evidence.

User question:
{state["question"]}

Web-search evidence:
{state.get("web_results", "")}

Return good if the evidence is directly relevant and sufficient
to answer the question.

Return weak if the results are empty, irrelevant, or insufficient.

Do not mark evidence good just because search results exist.

Return valid JSON:
{{"grade": "good"}}
or
{{"grade": "weak"}}
""")

    return {
        "web_grade": grade.grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.grade.upper()}",
        ),
    }


def after_web(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:

    if state["web_grade"] == "good":
        return "generate_from_web"

    if state.get("retry_count", 0) < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


# ---------------------------------------------------------
# 5. QUERY REWRITING
# ---------------------------------------------------------

def rewrite_query(state: AgentState):
    previous_query = (
        state.get("current_query")
        or state["question"]
    )

    rewritten = llm().invoke(f"""
Rewrite the following search query to improve retrieval.

Original user question:
{state["question"]}

Previous search query:
{previous_query}

Instructions:
- Preserve the user's original intent.
- Add useful keywords and relevant context.
- For current events, include terms that help find recent information.
- Do not answer the question.
- Return only the rewritten search query.
""").content.strip()

    if not rewritten:
        rewritten = previous_query

    new_retry_count = state.get("retry_count", 0) + 1

    return {
        "current_query": rewritten,
        "retry_count": new_retry_count,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}",
        ),
    }


# ---------------------------------------------------------
# 6. GENERATE ANSWER FROM PRIVATE KB
# ---------------------------------------------------------

def generate_from_kb(state: AgentState):
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in state.get("kb_docs", [])
    )

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

Answer ONLY from the private company knowledge base below.

Instructions:
- Be concise and actionable.
- If steps are present, present them clearly.
- Do not invent company policy details.
- If the evidence does not answer part of the question, clearly say so.
- Mention that the answer is based on the company's private knowledge base.

Question:
{state["question"]}

Private knowledge base:
{context}
""").content

    citations = []
    seen = set()

    for doc in state.get("kb_docs", []):
        source = doc.metadata.get("source", "Private KB")

        if source not in seen:
            seen.add(source)
            citations.append({
                "title": source.split("/")[-1],
                "url": "",
                "type": "private_kb",
            })

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }


# ---------------------------------------------------------
# 7. GENERATE ANSWER FROM TAVILY RESULTS
# ---------------------------------------------------------

def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are an AI assistant with access to external web-search results.

Answer the user's question using the web evidence provided below.

User question:
{state["question"]}

Web search results:
{state.get("web_results", "")}

Instructions:
1. Base factual claims on the supplied search results.
2. For questions asking for latest/current information, prioritize
   the newest relevant information in the results.
3. Include publication dates when they are available in the results.
4. Include source URLs from the supplied results in a Sources section.
5. Never invent URLs, publication dates, budgets, release dates, or facts.
6. If sources disagree, explain the disagreement.
7. If the results do not establish an answer, say that the information
   could not be verified from the available search results.
8. Clearly distinguish confirmed facts from reported or estimated figures.
9. Do not use your pretrained knowledge to pretend that old information
   is current.
10. If the question concerns company-managed IT systems, advise the user
    to validate external guidance with their IT team.

Return a clear, useful answer.
""").content

    return {
        "answer": answer,
        "source_used": "web_search",
        # Preserve the URLs collected from Tavily for the frontend.
        "citations": state.get("citations", []),
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH",
        ),
    }


# ---------------------------------------------------------
# 8. DIRECT ANSWER
# ---------------------------------------------------------

def direct_answer(state: AgentState):
    answer = llm().invoke(
        f"Respond briefly and naturally to: {state['question']}"
    ).content

    return {
        "answer": answer,
        "source_used": "direct",
        "citations": [],
        "trace": add_trace(
            state,
            "Direct response → no retrieval",
        ),
    }


# ---------------------------------------------------------
# 9. INSUFFICIENT EVIDENCE
# ---------------------------------------------------------

def insufficient(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough reliable evidence in the company "
            "knowledge base or external web search to answer confidently. "
            "Please try rephrasing your question or consult a trusted source."
        ),
        "source_used": "insufficient_evidence",
        "citations": state.get("citations", []),
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence",
        ),
    }


# ---------------------------------------------------------
# 10. BUILD LANGGRAPH WORKFLOW
# ---------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "get_current_datetime": get_current_datetime,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")

    # All three routes are now connected.
    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "search_web": "search_web",
            "direct_answer": "direct_answer",
            "get_current_datetime": "get_current_datetime",
        },
    )

    # Private KB path.
    graph.add_edge("retrieve_kb", "grade_kb")

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    # Tavily web-search path.
    graph.add_edge("search_web", "grade_web")

    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    # Retry path: rewritten query is tried against the KB first,
    # then falls back to Tavily if KB evidence remains insufficient.
    graph.add_edge("rewrite_query", "retrieve_kb")

    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    graph.add_edge("get_current_datetime", END)

    return graph.compile()


agent_graph = build_graph()


# ---------------------------------------------------------
# 11. CHAT ENTRY POINT
# ---------------------------------------------------------

def ask(question: str):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(initial)