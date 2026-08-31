import os
import time
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv

from db import create_user, get_user_id, get_watched_movie_ids, get_disliked_movie_ids
from recommender_engine import load_movies, load_embeddings
from chatbot_tools import build_tools, default_filters

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def classify_error(e):
    """Decide whether error is worth retrying (rate limit, timeout, transient server error) 
    or something that will fail identically every time (bad request, auth failure) and should 
    just be raised immediately instead of retried."""
    text = str(e)
    code = getattr(e, "code", None) or getattr(e, "status_code", None)

    if code == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "rate_limit"
    if code in (500, 502, 503, 504) or any(s in text for s in ("500", "502", "503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED")):
        return "transient"
    return "fatal"


# Tool schemas (what Gemini's allowed to call and w/ what arguments)
GET_RECOMMENDATIONS = types.FunctionDeclaration(
    name="get_recommendations",
    description="Find or refine movie recommendations. Filters accumulate across turns automatically — only pass the fields " 
                "the user actually just mentioned; do NOT re-send filters from earlier in the conversation, the system "
                "remembers them for you. Set reset=true ONLY when the user clearly wants to abandon the current search "
                "and start a completely different one (e.g. 'forget that, show me horror movies instead'), not for ordinary "
                "refinements like 'actually nothing after 2010'. The result includes 'active_filters_summary' showing the full " 
                "current filter state, and 'title_match' describing how an anchor title (if mentioned this turn) was resolved: " 
                "'exact', 'fuzzy' (confirm with the user before trusting it as the anchor), 'not_found', or 'no_query'."
                "The result may also include 'filter_validation' when a genre or actor you specified didn't exactly match the " \
                "dataset: 'corrected' lists (your_value, actual_value) pairs that were auto-corrected via fuzzy match — mention these "
                "briefly so the user knows what was actually used (e.g. 'I read that as \"science fiction\"'). 'unmatched' lists values " \
                "that don't exist in the dataset at all — tell the user plainly that filter wasn't applied, rather than letting "
                "them think zero results means nothing matches their taste.",
    parameters={
        "type": "object",
        "properties": {
            "target_movie": {"type": "string", "nullable": True,
                              "description": "A specific movie they liked, only if newly mentioned this turn."},
            "include_genres": {"type": "array", "items": {"type": "string"}},
            "exclude_genres": {"type": "array", "items": {"type": "string"}},
            "include_actors": {"type": "array", "items": {"type": "string"}},
            "exclude_actors": {"type": "array", "items": {"type": "string"}},
            "min_year": {"type": "integer", "nullable": True},
            "max_year": {"type": "integer", "nullable": True},
            "min_rating": {"type": "number", "nullable": True},
            "reset": {"type": "boolean", "nullable": True,
                      "description": "True only to wipe all prior filters and start a brand new search."},
        }
    }
)

MORE_RECOMMENDATIONS = types.FunctionDeclaration(
    name="more_recommendations",
    description="Continue the previous recommendation list with the next batch. "
                "Call this when the user asks for more, without repeating their original filters.",
    parameters={"type": "object", "properties": {}}
)

GET_MOVIE_DETAILS = types.FunctionDeclaration(
    name="get_movie_details",
    description="Get full details (overview, rating, cast, year) for one specific movie the user asks about.",
    parameters={"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
)

LOG_FEEDBACK = types.FunctionDeclaration(
    name="log_feedback",
    description="Record that the user watched, rated, or liked/disliked a movie — whether it came from "
                "a recommendation just shown, or a movie they mention on their own (e.g. logging something "
                "they watched with no recommendation involved).",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "watched": {"type": "boolean", "nullable": True},
            "liked": {"type": "boolean", "nullable": True},
            "rating": {"type": "number", "nullable": True}
        },
        "required": ["title"]
    }
)

TOOL = types.Tool(function_declarations=[GET_RECOMMENDATIONS, MORE_RECOMMENDATIONS, GET_MOVIE_DETAILS, LOG_FEEDBACK])

SYSTEM_INSTRUCTION = """You are a friendly, conversational movie recommendation assistant.

The user drives the conversation — don't force a fixed menu of options. They might want recommendations, want to look up 
a specific movie, want to log feedback on something they already watched (even if you never recommended it), or just chat.

Filters accumulate automatically across the conversation — when calling get_recommendations, only include the fields 
the user is mentioning right now. A follow-up like "actually nothing after 2010" should be called with ONLY max_year set while
still remembering what the user requested as filters in previous turns, not by re-specifying genres or actors from earlier turns. 
Check the 'active_filters_summary' in the tool's response if you want to confirm what's currently active before describing it 
to the user. Only set reset=true when the user is clearly abandoning the current search for an unrelated one.

Use the available tools whenever the user's message calls for one. After a tool returns results, summarize them naturally 
in your own words rather than dumping raw data.

IMPORTANT: when get_recommendations returns a 'title_match' field, check its status before you say anything about an anchor movie:
- 'exact': fine to proceed normally, no need to mention the matching process at all.
- 'fuzzy': you did NOT confidently find that movie. Say something like "I couldn't find an
  exact match for '<queried_title>' — did you mean '<matched_title>'?" and give the
  recommendations you already have (genre/filter-based this turn, not anchored), making
  clear they're not yet based on that specific movie.
- 'not_found': tell the user plainly you couldn't find that title, and that you're giving
  genre/filter-based picks instead.

IMPORTANT: if get_recommendations returns 'filter_validation', don't stay silent about it. For 'corrected' values, briefly confirm 
what you understood (e.g. "I matched 'scifi' to 'science fiction'"). For 'unmatched' values, clearly tell the user that 
specific genre or actor wasn't found in the dataset and wasn't applied as a filter — this matters because a result of 
"no recommendations found" should never be confused with "your filter was silently ignored."

Never imply a recommendation is "based on" a specific movie unless the match status was 'exact'.
"""


# Handle possible errors with Gemini API in chat
class ChatUnavailableError(Exception):
    """Raised when Gemini is unreachable after all retries and shows friendly message instead of crashing."""
    pass


MAX_RETRIES = 3
BASE_WAIT_SECONDS = 30


def call_gemini(history):
    """Calls generate_content with retry-with-backoff for rate limits/transient errors. Fatal errors 
    (bad request, bad API key) are raised immediately."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(
                model="gemini-3.6-flash",
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[TOOL],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                    )
                )
            )
        except ClientError as e:
            kind = classify_error(e)
            last_error = e

            if kind == "fatal":
                raise

            if attempt < MAX_RETRIES - 1:
                wait = BASE_WAIT_SECONDS * (attempt + 1)  
                print(f"(Having trouble reaching the model — retrying in {wait}s...)")
                time.sleep(wait)

    raise ChatUnavailableError(last_error)


def login():
    choice = input("New user or returning? (new/returning): ").strip().lower()
    if choice == "new":
        username = input("Choose a username: ").strip()
        user_id = create_user(username)
        while user_id is None:
            username = input("That username is taken. Choose another: ").strip()
            user_id = create_user(username)
    else:
        username = input("Enter your username: ").strip()
        user_id = get_user_id(username)
        while user_id is None:
            username = input("No account found with that name. Try again: ").strip()
            user_id = get_user_id(username)
    return user_id


def send_and_handle(history, tool_functions):
    """Sends convo to Gemini, executes any tool calls, feeds results back, and returns once Gemini produces plain-text reply."""
    while True:
        response = call_gemini(history)

        model_content = response.candidates[0].content
        history.append(model_content)

        function_calls = [p.function_call for p in model_content.parts if p.function_call]
        if not function_calls:
            return response.text    # plain conversational reply first

        result_parts = []
        for fc in function_calls:
            func = tool_functions.get(fc.name)
            result = func(**dict(fc.args)) if func else {"error": f"Unknown tool: {fc.name}"}
            result_parts.append(types.Part.from_function_response(name=fc.name, response=result))

        history.append(types.Content(role="user", parts=result_parts))
        # loop again for gemini to write actual reply


def main():
    user_id = login()

    session = {
        "user_id": user_id,
        "movies": load_movies(),
        "embeddings": load_embeddings(),
        "seen_ids": get_watched_movie_ids(user_id),
        "disliked_ids": get_disliked_movie_ids(user_id),
        "active_filters": default_filters(),
    }
    tool_functions = build_tools(session)

    print("\nHey! I'm your movie assistant. Ask me for recommendations, tell me about "
          "something you watched, or ask about a specific movie. Type 'quit' anytime to leave.\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye", "done", "goodbye", "stop", "leave"):
            print("Bot: See you next time!")
            break
        if not user_input:
            continue
        history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        try:
            reply = send_and_handle(history, tool_functions)
            print(f"Bot: {reply}\n")
        except ChatUnavailableError:
            print("Bot: Sorry, I'm having trouble connecting right now. Try sending your last message again in a minute.\n")
            history.pop()
        except ClientError as e:
            if classify_error(e) == "fatal":
                print("Bot: Something went wrong on my end that a retry won't fix. You may need to check your API key or request setup.\n")
                history.pop()
            else:
                raise



if __name__ == "__main__":
    main()