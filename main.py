import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)


# Define the structured response
class AgentResponse(BaseModel):
    answer: str


# Create structured model
structured_model = model.with_structured_output(AgentResponse)


while True:
    user_input = input("You: ")

    if user_input.lower() in ["q", "quit"]:
        print("Goodbye! 👋")
        break

    messages = [
        (
            "system",
            "You are a helpful AI assistant. "
            "Give clear and concise answers."
        ),
        (
            "human",
            user_input
        )
    ]

    try:
        response = structured_model.invoke(messages)
        print(f"Agent: {response.answer}")

    except Exception as e:
        print(f"Something went wrong: {e}")