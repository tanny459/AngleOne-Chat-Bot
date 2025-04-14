import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import markdownify
import time

BASE_URL = "https://www.angelone.in"
SUPPORT_URL = f"{BASE_URL}/support"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ScraperBot/1.0; +https://example.com/bot)"
}

def get_soup(url):
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def extract_support_links(soup):
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/support/") or href.startswith("https://www.angelone.in/support/"):
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)
    return list(links)

def scrape_page(url):
    print(f"Scraping: {url}")
    soup = get_soup(url)
    
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "No Title"

    # Find the FAQ section under `.list-content`
    faq_container = soup.find("div", class_="list-content")
    if not faq_container:
        return title, ""

    faqs = []
    tabs = faq_container.find_all("div", class_="tab")
    for tab in tabs:
        # Extract question
        label = tab.find("label", class_="tab-label")
        question = label.get_text(strip=True) if label else "No Question"

        # Extract answer
        answer_div = tab.find("div", class_="tab-content")
        if answer_div:
            # Flatten and clean paragraphs
            paragraphs = answer_div.find_all("p")
            answer_text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        else:
            answer_text = "No Answer"

        faqs.append(f"### Q: {question}\n**A:** {answer_text}\n")

    markdown_faqs = "\n\n".join(faqs)
    return title, markdown_faqs


def main():
    visited = set()
    to_visit = set([SUPPORT_URL])
    all_content = []

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
        try:
            soup = get_soup(current_url)
            title, markdown = scrape_page(current_url)
            all_content.append(f"# {title}\n\n**URL:** {current_url}\n\n{markdown}\n\n---\n")
            visited.add(current_url)
            new_links = extract_support_links(soup)
            to_visit.update(set(new_links) - visited)
            time.sleep(1)  # Be polite to the server
        except Exception as e:
            print(f"Failed to scrape {current_url}: {e}")

    with open("angelone_support_pages.txt", "w", encoding="utf-8") as f:
        f.writelines(all_content)

    print("Scraping completed. Content saved to 'angelone_support_pages.txt'.")

if __name__ == "__main__":
    main()
