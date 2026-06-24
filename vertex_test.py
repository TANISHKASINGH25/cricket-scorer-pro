import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(
    project="sportsanalytics-495612",
    location="us-central1"
)

models = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3.5-flash",
    "gemini-3.5-flash-001",
    "gemini-3.5-pro"
]

for model_name in models:
    try:
        model = GenerativeModel(model_name)
        response = model.generate_content("hello")
        print(f"✅ {model_name}")
    except Exception as e:
        print(f"❌ {model_name}: {e}")