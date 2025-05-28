import os
import random
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Example archetypes/analysis styles
STYLES = [
    "a pompous Ivy League psychologist",
    "a chaotic art critic who hates realism",
    "a conspiracy theorist with a podcast",
    "a dream interpreter who just read Freud",
    "a philosopher who recently discovered caffeine",
]


def build_prompt(detected_objects):
    """
    Takes a list of (label, confidence) tuples from YOLO,
    returns a prompt string for GPT.
    """
    style = random.choice(STYLES)

    # Convert object list to readable sentence
   # if not detected_objects:
      #  object_summary = "an unrecognizable blur of existential ambiguity"
   # else:
      #  object_summary = ", ".join([f"a {label} ({conf:.0%} sure)" for label, conf in detected_objects])
    object_summary = " ".join(detected_objects)
    return (
        f"You are {style}. You have been shown a photo that contains: {object_summary}.\n\n"
        "Write an absurdly deep, overly confident psychological analysis of the person who took this selfie or food photo. "
        "Include references to imaginary theories, use big words, and make bold claims with no evidence."
    )


def generate_analysis(detected_objects):
    """
    Generates a fake, absurd psychological analysis from detected image objects.
    """
    prompt = build_prompt(detected_objects)

    try:
        response = openai.responses.create(
            model="gpt-4.1",  # Or "gpt-3.5-turbo" if GPT-4 is not needed
            input = "You are a flamboyant and highly imaginative psychoanalyst" + prompt,
            temperature=0.95
            #max_tokens=400
        )
        return response.output_text
    except Exception as e:
        return f"Error generating analysis: {e}"

print(generate_analysis(["cat", "chair", "dining table","chair", "potted plant", "teddy bear", "cake"]))