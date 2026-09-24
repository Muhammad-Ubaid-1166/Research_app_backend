# fastapi_deep_research_ai.py
# Requires: pip install fastapi uvicorn beautifulsoup4 requests python-dotenv pydantic \
#           tavily-python langchain-groq langchain-core
import os
import asyncio
import json
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from bs4 import BeautifulSoup
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ---------------------- Logging ----------------------
# All raw provider errors (rate limits, bad requests, etc.) go here only.
# Nothing from these logs is ever included in an API response body.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deep_research")

# ---------------------- Load Environment ----------------------
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_API_KEYS = [
    k for k in [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ] if k
]
if not GROQ_API_KEYS:
    raise RuntimeError("No GROQ_API_KEY_1 / GROQ_API_KEY_2 / GROQ_API_KEY_3 set in .env")

# Models tried in order — falls to the next one on any error (rate limit,
# bad request, timeout, etc.), not just when a model is fully unavailable.
MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# Cache one ChatGroq client per (model, api_key) pair instead of rebuilding
# it on every call.
_llm_client_cache = {}

def _get_llm_client(model: str, api_key: str) -> ChatGroq:
    cache_key = (model, api_key)
    if cache_key not in _llm_client_cache:
        _llm_client_cache[cache_key] = ChatGroq(model=model, groq_api_key=api_key)
    return _llm_client_cache[cache_key]

async def call_llm_with_fallback(messages, structured_output=None):
    """Try every (model, api_key) combination in order until one succeeds.

    Outer loop = model, inner loop = API key. Every failure is logged
    server-side with full provider detail; callers only ever see a generic
    RuntimeError if every combination is exhausted — no raw error text
    reaches the API response.
    """
    last_exception = None
    for model in MODEL_FALLBACKS:
        for api_key in GROQ_API_KEYS:
            try:
                llm = _get_llm_client(model, api_key)
                if structured_output is not None:
                    llm = llm.with_structured_output(structured_output, method="json_schema")
                result = await llm.ainvoke(messages)
                if last_exception is not None:
                    logger.info(f"Recovered using model={model} key=...{api_key[-4:]}")
                return result
            except Exception as ex:
                last_exception = ex
                logger.warning(f"[LLM Error] model={model} key=...{api_key[-4:]} -> {ex}")
                continue
    logger.error(f"All Groq model/API-key combinations failed. Last error: {last_exception}")
    raise RuntimeError("All configured models are currently unavailable.")

# ---------------------- Tavily Search Tool ----------------------
async def tavily_search(query: str, retries: int = 3, delay: float = 2.0) -> List[dict]:
    if not TAVILY_API_KEY:
        raise RuntimeError("Search service is not configured.")
    for attempt in range(retries):
        try:
            response = await asyncio.to_thread(tavily_client.search, query=query, max_results=5)
            return response.get("results", [])
        except Exception as ex:
            logger.warning(f"[Search Error] Attempt {attempt+1}: {ex}")
            await asyncio.sleep(delay)
    return []

# ---------------------- URL Scraper Tool ----------------------
@tool
def url_scrape(url: str) -> str:
    """Scrape and return the visible text content of a web page at the given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text(separator=' ', strip=True)
        return text[:5000]
    except Exception as e:
        logger.warning(f"[Scrape Error] {url}: {e}")
        return "Failed to scrape this page."

# ---------------------- Agent Schemas ----------------------
class QueryResponse(BaseModel):
    queries: List[str]
    thoughts: str

class FollowUpDecisionResponse(BaseModel):
    should_follow_up: bool
    reasoning: str
    queries: List[str]

class SearchResult(BaseModel):
    title: str
    url: str
    summary: str

class FollowUpLog(BaseModel):
    iteration: int
    should_follow_up: bool
    reasoning: str
    next_queries: List[str]

# ---------------------- API Schemas ----------------------
DEFAULT_MAX_SEARCHES = 7

class ResearchRequest(BaseModel):
    query: str
    max_searches: int = DEFAULT_MAX_SEARCHES

class ResearchResponse(BaseModel):
    query: str
    thoughts: str
    search_results: List[SearchResult]
    followups: List[FollowUpLog]
    report: str

# ---------------------- Prompts ----------------------
QUERY_AGENT_PROMPT = """You are an expert research strategist who plans web searches before any research begins.

Given a research topic, think step by step about what a thorough researcher would need to know:
- What are the different angles, subtopics, or perspectives this topic touches?
- What terminology, named entities, or specific facts would sharpen a search?
- Where might the most authoritative or recent information live (news, official sources, academic, technical docs)?

Then produce exactly 3 search queries that are:
- Diverse: each query targets a distinct angle or subtopic, not variations of the same phrase.
- Unique: no two queries should be likely to return overlapping results.
- High-quality: specific and concrete rather than vague, using precise terms a search engine can act on.
- Concise: written as real search-engine queries (a few keywords or a short phrase), not full sentences.

In "thoughts", briefly explain your search strategy and why these 3 queries together give good topic coverage."""

FOLLOWUP_PROMPT = """You are a meticulous research analyst reviewing findings gathered so far to decide whether more research is needed.

Evaluate the findings against the original research question:
- Identify any important angle, subtopic, claim, or perspective that is still missing, thin, or unverified.
- Identify any contradictions between sources that deserve a targeted follow-up search.
- If the findings already give solid, well-rounded coverage of the topic, do not force unnecessary follow-up queries.

If follow-up is needed, propose new search queries that are:
- Diverse and unique: each targets a distinct gap, and none overlap with each other or with queries already used.
- Non-redundant: never repeat or lightly rephrase a query that has already been searched.
- Specific: precise enough to plausibly surface the missing information.

In "reasoning", state clearly what is missing or unresolved and why the proposed queries (or the decision to stop) follow from that. Set "should_follow_up" to false once coverage is genuinely sufficient."""

SYNTHESIS_PROMPT = """You are a professional research report writer producing a polished, well-organized markdown report.

Given the original research query and a set of source summaries, write a complete report that:
- Opens with a brief executive summary of the key findings.
- Includes a table of contents linking to each section.
- Is organized into clear, logically ordered headings and subheadings covering the different angles of the topic.
- Synthesizes information across sources rather than listing summaries one by one — group related points, note agreements, and flag any disagreements between sources.
- Cites sources inline (e.g. by title or domain) next to the claims they support, and lists all sources in a "References" section at the end.
- Uses precise, neutral, factual language and avoids unsupported speculation.
- Ends with a short "Limitations / Open Questions" section noting any gaps in the available research.

Write only the final markdown report — no meta-commentary about the writing process."""

SEARCH_PROMPT = """You are a precise technical summarizer condensing a single webpage into research-ready notes.

Given the page's title, URL, and scraped content, write a 2-3 paragraph summary that:
- Captures the main claims, facts, findings, or arguments — not the page's navigation, ads, or boilerplate.
- Preserves specific details that matter for research (names, numbers, dates, findings) rather than vague generalities.
- Notes explicitly if the content seems unrelated to the expected topic, outdated, or too thin to be useful.
- Stays neutral and factual — report what the source says without adding outside opinions or unstated inferences.

No fluff, no restating the title, no meta-commentary about the summarization itself — just the substantive content."""

# Direct scrape-then-summarize (no tool-calling loop): every search result
# always needs to be scraped and summarized, so there's no "decision" for an
# agent to make. This avoids Groq/Llama's occasional malformed tool-call
# output (tool_use_failed), and it's one LLM call instead of two.

async def _scrape_url_async(url: str) -> str:
    return await asyncio.to_thread(url_scrape.invoke, {"url": url})

async def summarize_url(title: str, url: str, retries: int = 2, delay: float = 2.0) -> str:
    scraped_text = await _scrape_url_async(url)
    for attempt in range(retries):
        try:
            result = await call_llm_with_fallback(
                [HumanMessage(
                    content=f"{SEARCH_PROMPT}\n\nTitle: {title}\nURL: {url}\n\nPage content:\n{scraped_text}"
                )]
            )
            return result.content
        except Exception as ex:
            logger.warning(f"[Summarize Error] Attempt {attempt+1}: {ex}")
            await asyncio.sleep(delay)
    return "Failed to summarize this page."

# ---------------------- Research Coordinator ----------------------
class ResearchCoordinator:
    def __init__(self, query: str, max_searches: int = DEFAULT_MAX_SEARCHES):
        self.query = query
        self.max_searches = max_searches
        self.search_results: List[SearchResult] = []
        self.followup_logs: List[FollowUpLog] = []
        self.iteration = 1
        self.search_count = 0
        self.thoughts = ""

    async def research(self) -> ResearchResponse:
        query_response = await call_llm_with_fallback(
            [HumanMessage(content=f"{QUERY_AGENT_PROMPT}\n\nResearch topic: {self.query}")],
            structured_output=QueryResponse,
        )
        self.thoughts = query_response.thoughts
        logger.info(f"Query agent thoughts: {self.thoughts}")

        queries = query_response.queries
        while self.search_count < self.max_searches:
            await self.perform_search(queries)
            if self.search_count >= self.max_searches:
                break
            decision = await self.evaluate_followup()
            if not decision.should_follow_up:
                break
            self.iteration += 1
            queries = decision.queries

        report = await self.synthesize_report()
        return ResearchResponse(
            query=self.query,
            thoughts=self.thoughts,
            search_results=self.search_results,
            followups=self.followup_logs,
            report=report,
        )

    async def perform_search(self, queries: List[str]):
        for query in queries:
            if self.search_count >= self.max_searches:
                logger.info(f"Reached the {self.max_searches}-search limit — stopping.")
                break
            self.search_count += 1
            logger.info(f"Searching ({self.search_count}/{self.max_searches}): {query}")
            results = await tavily_search(query)
            for result in results:
                title = result.get("title")
                url = result.get("url")  # Tavily uses "url" (SerpAPI used "link")
                if not title or not url:
                    continue
                summary = await summarize_url(title, url)
                self.search_results.append(SearchResult(title=title, url=url, summary=summary))

    async def evaluate_followup(self) -> FollowUpDecisionResponse:
        findings = f"Query: {self.query}\n\n"
        for r in self.search_results:
            findings += f"Title: {r.title}\nURL: {r.url}\nSummary: {r.summary}\n\n"
        try:
            response = await call_llm_with_fallback(
                [HumanMessage(content=f"{FOLLOWUP_PROMPT}\n\n{findings}")],
                structured_output=FollowUpDecisionResponse,
            )
        except RuntimeError as ex:
            # All model/key combos exhausted for this step — don't fail the
            # whole run, just stop the follow-up loop and synthesize from
            # what we already have.
            logger.warning(f"Follow-up evaluation unavailable ({ex}); using results gathered so far.")
            response = FollowUpDecisionResponse(
                should_follow_up=False,
                reasoning="Follow-up evaluation unavailable.",
                queries=[],
            )
        self.followup_logs.append(FollowUpLog(
            iteration=self.iteration,
            should_follow_up=response.should_follow_up,
            reasoning=response.reasoning,
            next_queries=response.queries,
        ))
        return response

    async def synthesize_report(self) -> str:
        full_text = f"Query: {self.query}\n\n"
        for r in self.search_results:
            full_text += f"Title: {r.title}\nURL: {r.url}\nSummary: {r.summary}\n\n"
        result = await call_llm_with_fallback(
            [HumanMessage(content=f"{SYNTHESIS_PROMPT}\n\n{full_text}")]
        )
        return result.content

    async def research_stream(self):
        """Async generator that yields SSE events during the research process."""
        query_response = await call_llm_with_fallback(
            [HumanMessage(content=f"{QUERY_AGENT_PROMPT}\n\nResearch topic: {self.query}")],
            structured_output=QueryResponse,
        )
        self.thoughts = query_response.thoughts
        logger.info(f"Query agent thoughts: {self.thoughts}")
        yield {"event": "thoughts", "data": {"thoughts": self.thoughts}}

        queries = query_response.queries
        while self.search_count < self.max_searches:
            for query in queries:
                if self.search_count >= self.max_searches:
                    logger.info(f"Reached the {self.max_searches}-search limit — stopping.")
                    break
                self.search_count += 1
                logger.info(f"Searching ({self.search_count}/{self.max_searches}): {query}")
                yield {"event": "progress", "data": {
                    "search_count": self.search_count,
                    "max_searches": self.max_searches,
                    "current_query": query,
                }}
                results = await tavily_search(query)
                for result in results:
                    title = result.get("title")
                    url = result.get("url")
                    if not title or not url:
                        continue
                    summary = await summarize_url(title, url)
                    sr = SearchResult(title=title, url=url, summary=summary)
                    self.search_results.append(sr)
                    yield {"event": "search_result", "data": sr.model_dump()}

            if self.search_count >= self.max_searches:
                break

            decision = await self.evaluate_followup()
            self.followup_logs.append(FollowUpLog(
                iteration=self.iteration,
                should_follow_up=decision.should_follow_up,
                reasoning=decision.reasoning,
                next_queries=decision.queries,
            ))
            yield {"event": "followup", "data": {
                "iteration": self.iteration,
                "should_follow_up": decision.should_follow_up,
                "reasoning": decision.reasoning,
                "next_queries": decision.queries,
            }}

            if not decision.should_follow_up:
                break

            self.iteration += 1
            queries = decision.queries

        report = await self.synthesize_report()
        response = ResearchResponse(
            query=self.query,
            thoughts=self.thoughts,
            search_results=self.search_results,
            followups=self.followup_logs,
            report=report,
        )
        yield {"event": "result", "data": response.model_dump()}

# ---------------------- FastAPI App ----------------------
app = FastAPI(title="Deep Research AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/research", response_model=ResearchResponse)
async def research_endpoint(payload: ResearchRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if payload.max_searches < 1:
        raise HTTPException(status_code=400, detail="max_searches must be at least 1.")
    coordinator = ResearchCoordinator(payload.query, max_searches=payload.max_searches)
    try:
        return await coordinator.research()
    except RuntimeError:
        # Full provider error detail is already in the server logs — the
        # client only ever gets this generic message.
        logger.error(f"Research run failed for query: {payload.query!r}")
        raise HTTPException(
            status_code=503,
            detail="Research service is temporarily unavailable. Please try again shortly.",
        )

@app.get("/research/stream")
async def research_stream_endpoint(query: str, max_searches: int = DEFAULT_MAX_SEARCHES):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if max_searches < 1:
        raise HTTPException(status_code=400, detail="max_searches must be at least 1.")

    coordinator = ResearchCoordinator(query, max_searches=max_searches)

    async def event_generator():
        try:
            async for event in coordinator.research_stream():
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        except RuntimeError as ex:
            logger.error(f"Research stream failed for query: {query!r}")
            yield f"event: error\ndata: {json.dumps({'detail': 'Research service is temporarily unavailable. Please try again shortly.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Run with: uvicorn fastapi_deep_research_ai:app --reload
