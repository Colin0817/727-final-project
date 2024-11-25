from pytz import UTC
import requests
import pandas as pd
from datetime import datetime
from dateutil import parser


def fetch_reviews_with_fixed_pages(
    app_id, keywords, start_date, end_date, output_csv, max_pages=20
):
    reviews = []

    for page in range(1, max_pages + 1):
        # URL
        url = f"https://itunes.apple.com/rss/customerreviews/page={page}/id={app_id}/sortBy=mostHelpful/json"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:117.0) Gecko/20100101 Firefox/117.0"
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"inaccessible page {page},error: {response.status_code}")
                break

            data = response.json()
            entries = data.get("feed", {}).get("entry", [])

            # if none, skip and continue
            if not entries:
                print(f" No comments relevant in {page} 。")
                continue

            print(f"retriving page {page} ,find {len(entries)} comments。")

            # for loop to parse each comment
            for entry in entries:
                try:
                    content = entry["content"]["label"]
                    date_str = entry["updated"]["label"]
                    date = parser.parse(date_str)
                    rating = entry["im:rating"]["label"]

                    # check keywords and date range
                    if (
                        any(keyword.lower() in content.lower() for keyword in keywords)
                        and start_date <= date <= end_date
                    ):
                        reviews.append(
                            {
                                "content": content,
                                "date": date.strftime("%Y-%m-%d %H:%M:%S"),
                                "rating": rating,
                            }
                        )
                except Exception as e:
                    print(f"Error parsing single comment:{e}")

        except requests.exceptions.RequestException as e:
            print(f"connention error:{e}")
            break
        except ValueError as e:
            print(f"JSON error:{e}")
            break

    # Save into CSV
    if not reviews:
        print("No comments found.")
        return

    df = pd.DataFrame(reviews)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Comments are saved in {output_csv}")


if __name__ == "__main__":
    # ID and other parameters
    app_id = "6473753684"
    keywords = [
        "plus",
        "pro",
        "claude pro",
        "upgrade",
        "premium",
        "claudepro",
    ]
    start_date = datetime.strptime("2023-10-01", "%Y-%m-%d").replace(tzinfo=UTC)
    end_date = datetime.strptime("2024-10-01", "%Y-%m-%d").replace(tzinfo=UTC)
    output_csv = "appstore_reviews_claude.csv"
    max_pages = 10

    fetch_reviews_with_fixed_pages(
        app_id, keywords, start_date, end_date, output_csv, max_pages
    )
