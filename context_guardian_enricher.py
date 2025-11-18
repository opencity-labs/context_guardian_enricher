from typing import Any, Dict, List, Set, Tuple, Optional, Union
from cat.mad_hatter.decorators import hook
from cat.log import log
from cat.looking_glass.stray_cat import StrayCat
from cat.convo.messages import CatMessage
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, ParseResult


def add_utm_tracking_to_url(url: str, utm_source: str) -> str:
    """
    Add UTM tracking parameter to a URL if it doesn't already have utm_source.
    
    Args:
        url: The URL to add UTM tracking to
        utm_source: The UTM source parameter value
        
    Returns:
        The URL with UTM tracking added (if applicable)
    """

    if not utm_source:  # Skip UTM tracking if utm_source is empty
        return url

    parsed: ParseResult = urlparse(url)
    query_params: Dict[str, List[str]] = parse_qs(parsed.query)

    if 'utm_source' not in query_params:
        query_params['utm_source'] = [utm_source]
        new_query: str = urlencode(query_params, doseq=True)
        parsed = parsed._replace(query=new_query)
        return urlunparse(parsed)
    
    return url


def enrich_links_with_utm(text: str, utm_source: str = "") -> str:
    """
    Find all HTTP/HTTPS URLs in text and add UTM tracking.
    Works with both markdown links and plain URLs.
    
    Args:
        text: The text containing URLs to be enriched
        utm_source: The UTM source parameter value
        
    Returns:
        The text with URLs enriched with UTM tracking
    """
    if not utm_source:  # Skip UTM tracking if utm_source is empty
        return text
    
    # Step 1: Transform plain text URLs to markdown format [url](url)
    # This regex matches URLs that are NOT already part of markdown links
    # Uses negative lookbehind to avoid URLs preceded by ]( and \S+ to match any non-whitespace URL
    plain_url_pattern: str = r'(?<!\]\()https?://\S+'
    
    def convert_to_markdown(match: re.Match[str]) -> str:
        url: str = match.group(0)
        return f'[{url}]({url})'
    
    # Convert plain URLs to markdown format
    text = re.sub(plain_url_pattern, convert_to_markdown, text)
    
    # Step 2: Add UTM tracking to URLs in markdown links
    def replace_url_in_markdown(match: re.Match[str]) -> str:
        url: str = match.group(0)
        return add_utm_tracking_to_url(url, utm_source)
    
    # This regex matches URLs inside markdown links (within parentheses after ])
    # Uses lookbehind (?<=\]\() and lookahead (?=\)) to ensure we're inside markdown link parentheses
    # [^)]+ matches any character except ) which is perfect since we're bounded by the assertions
    markdown_url_pattern: str = r'(?<=\]\()https?://[^)]+(?=\))'
    
    return re.sub(markdown_url_pattern, replace_url_in_markdown, text)

@hook
def fast_reply(_: Dict[str, Any], cat: StrayCat) -> Optional[CatMessage]:
    """
    Early interception of user query to decide whether to proceed with RAG+LLM.
    Can act as a panic button that always returns a default message.
    Also checks for procedural memories (tools/forms) before rejecting queries.
    
    Args:
        _: The input from the agent (unused)
        cat: The StrayCat instance
        
    Returns:
        CatMessage if no relevant context found or panic button is enabled, None otherwise
    """
    settings: Dict[str, Any] = cat.mad_hatter.get_plugin().load_settings()
    
    # Check if panic button is enabled - if so, return immediately with panic text
    if settings.get('panic_button_enabled', False):
        cat.recall_relevant_memories_to_working_memory()
        return CatMessage(user_id=cat.user_id, text=settings.get('panic_button_text', "Sorry, I'm under maintenance right now. Please try again later."))

    # return default message if the length of the user query is less than minimum length
    min_query_length: int = settings.get('min_query_length', 10)
    user_query: str = cat.working_memory.user_message_json.text.strip()
    if len(user_query) < min_query_length:
        return CatMessage(user_id=cat.user_id, text=settings.get('default_message', 'Sorry, I can\'t help you.'))

    # Regular source enricher behavior
    cat.recall_relevant_memories_to_working_memory()

    # Check if we have relevant context from declarative memories
    has_declarative_context: bool = bool(cat.working_memory.declarative_memories)
    
    # Check if user is currently in a form session
    form_ongoing: bool = False
    
    if hasattr(cat.working_memory, 'active_form'):
        form_ongoing = cat.working_memory.active_form is not None

    if not has_declarative_context and not form_ongoing:
        return CatMessage(user_id=cat.user_id, text=settings.get('default_message', 'Sorry, I can\'t help you.'))

    return None

@hook
def before_cat_reads_message(user_message_json: Dict[str, Any], cat: StrayCat) -> Dict[str, Any]:
    """
    Hook to modify user message before cat reads it.
    Appends current time to the user message.
    
    Args:
        user_message_json: The user message JSON object
        _: The StrayCat instance (unused)
        
    Returns:
        Modified user message JSON object
    """
    # append "current time" to user message
    from datetime import datetime
    current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user_message_json.text += f"\n\nCurrent time: {current_time}"
    return user_message_json


@hook
def before_cat_sends_message(message: CatMessage, cat: StrayCat) -> CatMessage:
    """
    Enrich the outgoing message with the sources used during the main (and optional double) pass.
    
    Args:
        message: The CatMessage to be sent
        cat: The StrayCat instance
        
    Returns:
        The enriched CatMessage with sources and UTM tracking
    """
    
    # if form_ongoing: # skip rejection if user is in a form session
    form_ongoing: bool = False
    if hasattr(cat.working_memory, 'active_form'):
        form_ongoing = cat.working_memory.active_form is not None
    if form_ongoing:
        log.debug("User is in a form session, skipping source enrichment")
        return message
    if "<no_sources>" in message.text:
        log.debug("No sources required.")
        message.text = message.text.replace("<no_sources>", "")
        return message
    
    settings: Dict[str, Any] = cat.mad_hatter.get_plugin().load_settings()
    utm_source: str = settings.get('utm_source', '')

    # Collect sources from declarative memories in order (most relevant first)
    sources: List[Dict[str, str]] = []
    seen_sources: Set[str] = set()
    for mem in cat.working_memory.declarative_memories:
        doc = mem[0]  # Document is the first element in the tuple
        source: Optional[str] = doc.metadata.get('source')
        title: Optional[str] = doc.metadata.get('title')  # Page title is available here for web pages
        if source and source not in seen_sources:
            sources.append({"url": source, "label": title or ""})
            seen_sources.add(source)

    if settings.get('double_pass', False):
        # Double pass: query memory with user query + generated response
        combined_text: str = cat.working_memory.user_message_json.text + " " + message.text
        embedding = cat.embedder.embed_query(combined_text)
        second_memories = cat.memory.vectors.declarative.recall_memories_from_embedding(embedding)

        second_sources: List[Dict[str, str]] = []
        seen_second_sources: Set[str] = set()
        for mem in second_memories:
            doc = mem[0]
            source: Optional[str] = doc.metadata.get('source')
            title: Optional[str] = doc.metadata.get('title')
            if source and source not in seen_second_sources:
                second_sources.append({"url": source, "label": title or ""})
                seen_second_sources.add(source)

        # Find intersection while preserving order from main sources
        relevant_sources: List[Dict[str, str]] = [s for s in sources if s['url'] in seen_second_sources]
        if not relevant_sources:
            relevant_sources = sources if sources else second_sources

    else:
        # Single pass: use all sources from main pass
        relevant_sources: List[Dict[str, str]] = sources

    message.sources = [{"url": add_utm_tracking_to_url(s['url'], utm_source), "label": s['label'].split("/")[0]} for s in relevant_sources] if relevant_sources else []

    # Add UTM tracking to all links in the final message
    message.text = enrich_links_with_utm(message.text, utm_source)
    return message