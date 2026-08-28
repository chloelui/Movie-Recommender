import os
import time
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv

from db import create_user, get_user_id, get_watched_movie_ids, get_disliked_movie_ids
from recommender_engine import load_movies, load_embeddings
from chatbot_tools import build_tools

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Tool schemas (what Gemini's allowed to call and w/ what arguments)
GET_RECOMMENDATIONS = types.FunctionDeclaration(
    name="get_recommendations",
    description="Find movie recommendations. Call this whenever the user wants something to watch, "
                "whether or not they gave a specific anchor movie, genre, actor, year, or rating. "
                "The result includes a 'title_match' field describing how the anchor title (if any) "
                "was resolved: 'exact' (confident match, safe to treat as the anchor), 'fuzzy' "
                "(only a guessed match was found — you MUST tell the user what you guessed and ask "
                "them to confirm before treating it as the anchor on a future turn), 'not_found' "
                "(no match at all — tell the user plainly and offer genre/filter-based picks instead), "
                "or 'no_query' (user didn't name a movie, nothing to report).",
    parameters={
        "type": "object",
        "properties": {
            "target_movie": {"type": "string", "nullable": True,
                              "description": "A specific movie they liked, if mentioned. Otherwise omit."},
            "include_genres": {"type": "array", "items": {"type": "string"}},
            "exclude_genres": {"type": "array", "items": {"type": "string"}},
            "include_actors": {"type": "array", "items": {"type": "string"}},
            "exclude_actors": {"type": "array", "items": {"type": "string"}},
            "min_year": {"type": "integer", "nullable": True},
            "max_year": {"type": "integer", "nullable": True},
            "min_rating": {"type": "number", "nullable": True},
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

The user drives the conversation — don't force a fixed menu of options. They might want
recommendations, want to look up a specific movie, want to log feedback on something they
already watched (even if you never recommended it), or just chat.

Use the available tools whenever the user's message calls for one. After a tool returns
results, summarize them naturally in your own words rather than dumping raw data.

IMPORTANT: when get_recommendations returns a 'title_match' field, check its status before
you say anything about an anchor movie:
- 'exact': fine to proceed normally, no need to mention the matching process at all.
- 'fuzzy': you did NOT confidently find that movie. Say something like "I couldn't find an
  exact match for '<queried_title>' — did you mean '<matched_title>'?" and give the
  recommendations you already have (which are genre/filter-based this turn, not anchored),
  making clear they're not yet based on that specific movie. Ask the user to confirm before
  you use it as an anchor going forward.
- 'not_found': tell the user plainly you couldn't find that title in your dataset, and that
  you're giving genre/filter-based picks instead.
- 'no_query': say nothing about title matching.

Never imply a recommendation is "based on" a specific movie unless the match status was 'exact'.
"""


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
    """Sends convo to Gemini, executes any tool calls, feeds results back, 
    and returns once Gemini produces plain-text reply."""
    while True:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[TOOL],
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO"))
            )
        )

        model_content = response.candidates[0].content
        history.append(model_content)

        function_calls = [p.function_call for p in model_content.parts if p.function_call]
        if not function_calls:
            return response.text    # plain conversational reply - done for this turn

        result_parts = []
        for fc in function_calls:
            func = tool_functions.get(fc.name)
            result = func(**dict(fc.args)) if func else {"error": f"Unknown tool: {fc.name}"}
            result_parts.append(types.Part.from_function_response(name=fc.name, response=result))

        history.append(types.Content(role="user", parts=result_parts))
        # loop again — Gemini now sees tool's result and writes actual reply


def main():
    user_id = login()

    session = {
        "user_id": user_id,
        "movies": load_movies(),
        "embeddings": load_embeddings(),
        "seen_ids": get_watched_movie_ids(user_id),
        "disliked_ids": get_disliked_movie_ids(user_id),
    }
    tool_functions = build_tools(session)

    print("\nHey! I'm your movie assistant. Ask me for recommendations, tell me about "
          "something you watched, or ask about a specific movie. Type 'quit' anytime to leave.\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "bye", "stop"):
            print("Bot: See you next time!")
            break
        if not user_input:
            continue

        history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
        reply = send_and_handle(history, tool_functions)
        print(f"Bot: {reply}\n")



if __name__ == "__main__":
    main()