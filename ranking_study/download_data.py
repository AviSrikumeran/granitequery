"""Download GSM8K and Alpaca into ./data."""
import urllib.request
from pathlib import Path

FILES = {
    "gsm8k_train.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "alpaca.json": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
}

out = Path("data")
out.mkdir(exist_ok=True)
for name, url in FILES.items():
    print("downloading", name)
    urllib.request.urlretrieve(url, out / name)
print("done")
