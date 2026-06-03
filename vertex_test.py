import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
vertexai.init(
    project="sportsanalytics-495612",
    location="us-central1"
)

# Load Gemini model
model = GenerativeModel("gemini-2.5-flash")

# Send test prompt
response = model.generate_content("Say hello from Vertex AI in one sentence")

print(response.text)