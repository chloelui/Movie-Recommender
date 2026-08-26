import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define parameters to parse
extract_preferences_decl = types.FunctionDeclaration(
    name="extract_preferences",
    description="Extract structured movie search preferences from a user's natural-language request. "
                "Not all fields will be mentioned — leave anything not mentioned as null or empty.",
    parameters={
        "type": "object",
        "properties": {
            "target_movie": {
                "type": "string",
                "nullable": True,
                "description": "A specific movie the user says they liked or wants similar picks to. "
                                "Null if the user only describes a mood/genre/vibe with no specific movie."
            },
            "include_genres": {"type": "array", "items": {"type": "string"}},
            "exclude_genres": {"type": "array", "items": {"type": "string"}},
            "include_actors": {"type": "array", "items": {"type": "string"}},
            "exclude_actors": {"type": "array", "items": {"type": "string"}},
            "min_year": {"type": "integer", "nullable": True},
            "max_year": {"type": "integer", "nullable": True},
            "min_rating": {"type": "number", "nullable": True}
        },
        "required": ["include_genres", "exclude_genres", "include_actors", "exclude_actors"]
    }
)

tool = types.Tool(function_declarations=[extract_preferences_decl])


# Ask LLM to parse user's input
def parse_preferences(user_text):
    """Turn free-form text into the structured filter dict recommender.py expects."""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_text,
        config=types.GenerateContentConfig(
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["extract_preferences"]
                )
            )
        )
    )
    call = response.candidates[0].content.parts[0].function_call
    args = dict(call.args)

    # Ensure every key exists even if Gemini ommitted empty field
    defaults = {
        "target_movie": None, "include_genres": [], "exclude_genres": [],
        "include_actors": [], "exclude_actors": [],
        "min_year": None, "max_year": None, "min_rating": None
    }
    defaults.update(args)
    return defaults