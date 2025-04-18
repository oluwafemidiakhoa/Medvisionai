
import os

print("Checking environment variables...")
print(f"GEMINI_API_KEY exists: {bool(os.environ.get('GEMINI_API_KEY'))}")
print(f"HF_API_TOKEN exists: {bool(os.environ.get('HF_API_TOKEN'))}")
print(f"UMLS_APIKEY exists: {bool(os.environ.get('UMLS_APIKEY'))}")

# If keys exist, show first few characters to confirm they're loaded correctly
for key in ['GEMINI_API_KEY', 'HF_API_TOKEN', 'UMLS_APIKEY']:
    value = os.environ.get(key)
    if value:
        print(f"{key}: {value[:5]}...")
