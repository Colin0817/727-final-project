from google_play_scraper import Sort, reviews
from datetime import datetime
import pandas as pd

# App ID
app_id = "com.anthropic.claude"

# keywords
keywords = [
    "plus",
    "pro",
    "claude pro",
    "upgrade",
    "premium",
    "claudepro",
    "chatgptplus",
    "gptplus",
    "gpt4",
]

# time range
start_date = datetime(2023, 10, 1)
end_date = datetime(2024, 10, 1)

# language and country
languages = ["en"]
countries = [
    "us",
    "gb",
    "ca",
    "au",
]

all_reviews = []

# max pages
max_pages = 20

# retrieve reviews
for lang in languages:
    for country in countries:
        continuation_token = None
        current_page = 0

        while current_page < max_pages:
            try:
                results, continuation_token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.MOST_RELEVANT,
                    count=100,
                    continuation_token=continuation_token,
                )

                # filter reviews
                filtered_reviews = [
                    review
                    for review in results
                    if start_date <= review["at"] <= end_date
                    and any(
                        keyword.lower() in review["content"].lower()
                        for keyword in keywords
                    )
                ]
                all_reviews.extend(filtered_reviews)

                current_page += 1

                # print progress
                print(
                    f"language: {lang}, country: {country}, page(now): {current_page}"
                )

                # continuation_token = None，end of reviews
                if not continuation_token:
                    print(
                        f"End of comment crawl, reached the last page (lang={lang}, country={country})"
                    )
                    break

            except Exception as e:
                print(f"Error fetching reviews for lang={lang}, country={country}: {e}")
                break

# DataFrame
df = pd.DataFrame(all_reviews)

# save to CSV
output_file = "filtered_reviews_claude.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"comments are saved in {output_file} !")
