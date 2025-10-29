import sys
import json
import time
import os
import yaml
from litellm import completion
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------- Config -------
COUNTRIES = [
    "India",
    "China",
    "United States",
    "Indonesia",
    "Pakistan",
    "Nigeria",
    "Brazil",
    "Bangladesh",
    "Russia",
    "Mexico",
    "Ethiopia",
    "Japan",
    "Philippines",
    "Egypt",
    "DR Congo",
    "Vietnam",
    "Iran",
    "Turkey",
    "Germany",
    "Thailand",
    "United Kingdom",
    "Tanzania",
    "France",
    "South Africa",
    "Italy",
    "Kenya",
    "Myanmar",
    "Colombia",
    "South Korea",
    "Sudan",
    "Yemen",
    "Saudi Arabia",
    "North Korea",
    "Taiwan",
    "Romania",
    "Czechia",
    "Tajikistan",
    "Hungary",
    "Turkmenistan",
    "Kyrgyzstan",
    "Finland",
    "Norway",
    "Costa Rica",
    "Kuwait",
    "Panama",
    "Botswana",
    "Trinidad and Tobago",
    "Eswatini",
    "Djibouti",
    "Maldives"
]

PROMPT_TEMPLATE = """
Task: For the country I provide, return a 10-dimensional feature vector.
Score each dimension from 1 to 100, where:
- 1 = extremely poor outcome / worst case seen globally
- 100 = excellent outcome / best case seen globally

Scoring meanings for each dimension:
- life_expectancy: 1 = extremely low lifespan, 100 = extremely high lifespan
- under5_mortality: 1 = very high child mortality, 100 = extremely low child mortality
- disease_burden: 1 = very high disease burden, 100 = very low disease burden
- uhc_coverage: 1 = almost no access to essential healthcare, 100 = universal, high-quality access
- health_spending: 1 = extremely low healthcare spending per person, 100 = extremely high
- gdp_per_capita: 1 = extremely low national income, 100 = extremely high
- poverty_rate: 1 = extreme poverty widespread, 100 = almost no poverty
- inequality: 1 = extremely unequal income distribution, 100 = very equal
- education_level: 1 = very low average education, 100 = very high
- conflict_disaster_risk: 1 = very high conflict/disaster risk, 100 = very safe and stable

Return the result as a JSON object only, with these exact keys:
{
  "life_expectancy": ...,
  "under5_mortality": ...,
  "disease_burden": ...,
  "uhc_coverage": ...,
  "health_spending": ...,
  "gdp_per_capita": ...,
  "poverty_rate": ...,
  "inequality": ...,
  "education_level": ...,
  "conflict_disaster_risk": ...
}

Country: {country}

Output only the JSON. No explanations.
""".strip()
# -----------------------

def short_name(model_name: str) -> str:
    """Create a short safe identifier for filename."""
    return model_name.replace("/", "_").replace(":", "_").replace(".", "_")

def extract_json(content: str):
    """Try to clean and parse JSON from model output."""
    content = content.strip()

    # Remove code fences if present
    if content.startswith("```"):
        content = content.strip("`")
        # remove json or similar language tag if present
        content = content.replace("json", "", 1).strip()

    return json.loads(content)

def query_with_retries(model_name: str, prompt: str, retries: int = 3, delay: float = 1.5):
    """Query model and retry if JSON parsing fails."""
    for attempt in range(1, retries + 1):
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0
        )
        content = response["choices"][0]["message"]["content"]

        try:
            return extract_json(content)
        except Exception:
            print(f"[WARN] JSON parse failed (attempt {attempt}/{retries}).")
            print("Output was:\n", content)

            if attempt < retries:
                print("Retrying...")
                time.sleep(delay)

    return None  # All attempts failed

def load_models_yaml():
    """Load models.yaml and return the models dictionary."""
    # models.yaml is in utility_analysis/ directory
    # This file is in utility_analysis/experiments/decision_modeling/
    # So we need to go up 3 levels: decision_modeling -> experiments -> utility_analysis
    utility_analysis_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_yaml_path = os.path.join(utility_analysis_dir, 'models.yaml')
    with open(models_yaml_path, 'r') as f:
        return yaml.safe_load(f)

def resolve_model_name(input_model: str) -> tuple[str, str]:
    """
    Resolve model name from models.yaml if it's a short name.
    Returns (full_model_name, short_model_name).
    If not found in models.yaml, returns (input_model, input_model).
    """
    models_config = load_models_yaml()
    
    if input_model in models_config:
        model_config = models_config[input_model]
        full_model_name = model_config.get('model_name', input_model)
        return full_model_name, input_model
    else:
        # Not found in models.yaml, use as-is
        return input_model, input_model

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_country_features.py <model_name_or_short_name>")
        sys.exit(1)

    input_model = sys.argv[1]
    full_model_name, short_model_name = resolve_model_name(input_model)
    
    outfile = f"country_features_{short_name(short_model_name)}.jsonl"

    print(f"Running with model: {full_model_name}")
    if input_model != full_model_name:
        print(f"Short name: {short_model_name}")
    print(f"Writing to: {outfile}")

    with open(outfile, "a", encoding="utf-8") as f:
        for country in COUNTRIES:
            # Avoid str.format so JSON braces in the template aren't treated as placeholders
            prompt = PROMPT_TEMPLATE.replace("{country}", country)

            features = query_with_retries(full_model_name, prompt)

            if features is None:
                print(f"[ERROR] Skipping {country} after repeated failures.")
                continue

            record = {
                "country": country,
                "model": full_model_name,
                #"model_short": short_model_name,
                "features": features
            }

            f.write(json.dumps(record) + "\n")
            print(f"✓ {country} processed")

    print("Done.")

if __name__ == "__main__":
    main()
