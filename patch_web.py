import requests
from bs4 import BeautifulSoup

url = "https://wini.com/franchise/"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
soup = BeautifulSoup(response.content, 'html.parser')

for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
    element.decompose()

sections = []
current_section = "Website Intro"
current_text = []

for element in soup.body.descendants:
    if getattr(element, 'name', None) in ['h1', 'h2', 'h3']:
        if current_text:
            text = ' '.join(' '.join(current_text).split())
            if text:
                sections.append((current_section, text))
        current_section = element.get_text(separator=' ', strip=True)
        current_text = []
    elif isinstance(element, str):
        if element.strip() and element.parent.name not in ['script', 'style']:
            current_text.append(element.strip())

if current_text:
    text = ' '.join(' '.join(current_text).split())
    if text:
        sections.append((current_section, text))

for s in sections[:3]:
    print("---")
    print("SECTION:", s[0])
    print("TEXT:", s[1][:100])
