import os
import re
import time
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse
import docx

def get_workspace_paths():
    """Determines the absolute path to the data folder relative to this script's location."""
    # File path: win-franchise-agent/rag/ingestion/web_scrapper_script.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(root_dir, "data")
    return data_dir

def normalize_url(url: str) -> str:
    """Strips query parameters and fragments to deduplicate page hits."""
    parsed = urlparse(url)
    # Rebuild URL without query parameters or trailing hashes
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), '', '', ''))

def parse_sitemap(sitemap_path: str) -> list:
    """Extracts and deduplicates absolute URLs from the sitemap XML file."""
    if not os.path.exists(sitemap_path):
        print(f"Error: Sitemap file not found at {sitemap_path}")
        return []
    
    print(f"Reading sitemap at: {sitemap_path}")
    try:
        tree = ET.parse(sitemap_path)
        root = tree.getroot()
        # Standard sitemap XML namespace
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        raw_urls = []
        for url_tag in root.findall('ns:url', namespace):
            loc = url_tag.find('ns:loc', namespace)
            if loc is not None and loc.text:
                raw_urls.append(loc.text.strip())
    except Exception as e:
        print(f"XML parsing warning: {e}. Falling back to regex parsing...")
        try:
            with open(sitemap_path, 'r', encoding='utf-8') as f:
                content = f.read()
            raw_urls = re.findall(r'<loc>(.*?)</loc>', content)
        except Exception as e2:
            print(f"Regex parsing failed: {e2}")
            return []

    # Deduplicate URLs based on base path normalization
    seen = set()
    deduplicated_urls = []
    for url in raw_urls:
        norm = normalize_url(url)
        if norm not in seen:
            seen.add(norm)
            deduplicated_urls.append(url)
            
    return deduplicated_urls

def format_html_table_to_markdown(table_tag) -> str:
    """Parses a BeautifulSoup table tag and renders a valid Markdown table."""
    rows = []
    max_cols = 0
    
    for tr in table_tag.find_all('tr', recursive=True):
        cells = []
        # Search only direct cells to avoid merged nested confusion
        for td in tr.find_all(['td', 'th'], recursive=False):
            # Format cell text nicely: remove line breaks, escape markdown pipes
            cell_text = ' '.join(td.get_text(separator=' ', strip=True).split()).replace('|', '\\|')
            cells.append(cell_text)
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
            max_cols = max(max_cols, len(cells))
            
    if not rows:
        return ""
        
    # Create standard header divider e.g., | --- | --- |
    header_divider = "| " + " | ".join(["---"] * max_cols) + " |"
    
    # Inject separator after the first row (which acts as the table header)
    rows.insert(1, header_divider)
    return "\n".join(rows)

def normalize_custom_page_elements(soup: BeautifulSoup):
    """
    Identifies custom, non-standard Elementor layout widgets (e.g., pseudo-tables
    built with nested columns or custom lists with direct text siblings) and 
    normalizes them into standard semantic HTML tags (like <table> and <p>) so 
    the existing general parser handles them naturally and robustly.
    """
    # 1. Normalize Win Advantage items by wrapping description texts in pristine <p> tags
    for item in soup.find_all(class_='advantage-list-item'):
        container = item.find(class_='elementor-widget-container')
        if container:
            h3 = container.find('h3')
            if h3:
                desc_parts = []
                # Accumulate text from all following siblings in the container
                for sibling in list(h3.next_siblings):
                    if hasattr(sibling, 'get_text'):
                        txt = sibling.get_text(separator=' ', strip=True)
                    else:
                        txt = str(sibling).strip()
                    if txt:
                        desc_parts.append(txt)
                    sibling.extract() # Remove so they aren't processed in duplicates
                
                full_desc = ' '.join(' '.join(desc_parts).split())
                if full_desc:
                    new_p = soup.new_tag('p')
                    new_p.string = full_desc
                    h3.insert_after(new_p)

    # 2. Convert "Type of Expenditure" Elementor pseudo-table to a real <table>
    exp_containers = soup.find_all(class_='expenditure-text')
    if exp_containers:
        table_rows_html = []
        for exp in exp_containers:
            # 2a. Extract label title
            title_node = exp.find(class_='underlined-text')
            title = title_node.get_text(separator=' ', strip=True) if title_node else exp.get_text(separator=' ', strip=True)
            title = ' '.join(title.split())
            
            # 2b. Extract associated pop-up tooltip description
            parent_widget = exp.find_parent(class_='elementor-widget-container')
            description = ""
            if parent_widget:
                modal_body = parent_widget.find(class_=['modal-body-2', 'modal-body'])
                if modal_body:
                    description = ' '.join(modal_body.get_text(separator=' ', strip=True).split())
            
            # 2c. Extract amount by scanning forward in sequential DOM flow for the next price widget
            amount = ""
            scan_node = exp
            while scan_node:
                scan_node = scan_node.next_element
                if not scan_node:
                    break
                # Break scan if we bump into the next label to prevent bleed-over
                if hasattr(scan_node, 'get') and scan_node.get('class') and 'expenditure-text' in scan_node.get('class', []):
                    break
                # Identify next heading container holding a currency token
                if hasattr(scan_node, 'get') and scan_node.get('class') and 'elementor-heading-title' in scan_node.get('class', []):
                    txt = ' '.join(scan_node.get_text(strip=True).split())
                    if '$' in txt:
                        amount = txt
                        break
            
            if title and (amount or description):
                table_rows_html.append(f"<tr><td>{title}</td><td>{amount}</td><td>{description}</td></tr>")
                
        if table_rows_html:
            # Attempt to inject the new real table after the target heading
            target_header = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3', 'h4'] and 'investment information' in tag.get_text().lower())
            if target_header:
                table_str = (
                    "<table>"
                    "<thead><tr><th>Type of Expenditure</th><th>Amount</th><th>Description</th></tr></thead>"
                    "<tbody>" + "".join(table_rows_html) + "</tbody></table>"
                )
                new_table_soup = BeautifulSoup(table_str, 'html.parser')
                target_header.insert_after(new_table_soup)

def scrape_web_page(url: str) -> list:
    """
    Fetches page, discards noise, and parses text, lists, and tables 
    into structured hierarchical sections/subsections.
    """
    print(f"Scraping web content: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # Safe timeout to prevent hang
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to retrieve {url}: {e}")
        return []
        
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Run DOM normalizer to convert custom/complex sales layouts to standard tags
    normalize_custom_page_elements(soup)
    
    # Strip navigation, headers, footers, and scripting code
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "form", "noscript"]):
        element.decompose()
        
    blocks = []
    current_section = "Intro"
    current_subsection = ""
    current_content = []
    
    def flush_block():
        nonlocal current_content
        if current_content:
            body = "\n\n".join(current_content).strip()
            if body:
                blocks.append({
                    "source": "website",
                    "url_or_path": url,
                    "section": current_section,
                    "subsection": current_subsection,
                    "text": body
                })
            current_content = []

    # Identify target tags in the main body
    body_root = soup.body if soup.body else soup
    target_tags = body_root.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'table', 'ul', 'ol'])
    
    # Set tracking to ensure children aren't processed twice
    processed_tags = set()
    
    for tag in target_tags:
        if tag in processed_tags:
            continue
            
        # Add all descendants of this element to avoid duplicates
        for desc in tag.descendants:
            processed_tags.add(desc)
            
        if tag.name in ['h1', 'h2']:
            flush_block()
            current_section = ' '.join(tag.get_text(separator=' ', strip=True).split())
            current_subsection = ""
        elif tag.name in ['h3', 'h4']:
            flush_block()
            current_subsection = ' '.join(tag.get_text(separator=' ', strip=True).split())
        elif tag.name == 'table':
            tbl_md = format_html_table_to_markdown(tag)
            if tbl_md:
                current_content.append(tbl_md)
        elif tag.name in ['ul', 'ol']:
            items = []
            for li in tag.find_all('li', recursive=False):
                txt = ' '.join(li.get_text(separator=' ', strip=True).split())
                if txt:
                    items.append(f"- {txt}")
            if items:
                current_content.append("\n".join(items))
        elif tag.name == 'p':
            txt = ' '.join(tag.get_text(separator=' ', strip=True).split())
            # Eliminate extremely short strings which are usually noise
            if txt and len(txt) > 10:
                current_content.append(txt)
                
    flush_block()
    return blocks

def format_docx_table_to_markdown(table) -> str:
    """Converts python-docx Table object to Markdown table text."""
    markdown_rows = []
    col_count = 0
    
    for idx, row in enumerate(table.rows):
        # Gather all unique cell text values (handling duplicate references for merged cells gracefully)
        cells = []
        for cell in row.cells:
            txt = cell.text.strip().replace('\n', ' ').replace('|', '\\|')
            cells.append(txt)
            
        if not cells:
            continue
            
        # Formulate Markdown row
        row_string = "| " + " | ".join(cells) + " |"
        markdown_rows.append(row_string)
        
        if idx == 0:
            col_count = len(cells)
            
    if not markdown_rows:
        return ""
        
    # Create markdown divider row
    divider = "| " + " | ".join(["---"] * col_count) + " |"
    markdown_rows.insert(1, divider)
    
    return "\n".join(markdown_rows)

def parse_docx(docx_path: str) -> list:
    """Reads docx and maintains sequence ordering of tables and paragraphs."""
    print(f"\nParsing Document: {docx_path}")
    if not os.path.exists(docx_path):
        print(f"Error: Document not found: {docx_path}")
        return []
        
    try:
        doc = docx.Document(docx_path)
    except Exception as e:
        print(f"Failed to read DOCX: {e}")
        return []
        
    blocks = []
    current_section = "Document Title"
    current_subsection = ""
    current_content = []
    
    def flush_block():
        nonlocal current_content
        if current_content:
            body = "\n\n".join(current_content).strip()
            if body:
                blocks.append({
                    "source": "document",
                    "url_or_path": os.path.basename(docx_path),
                    "section": current_section,
                    "subsection": current_subsection,
                    "text": body
                })
            current_content = []

    # We must preserve linear document order between paragraphs and tables.
    # We can inspect doc.element.body to get OXML elements in sequential order.
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    
    for child in doc.element.body:
        if isinstance(child, CT_P):
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
                
            # FDD Specific Layout Parser Heuristics
            # ITEM section headings usually signal key thematic topics
            if text.startswith("ITEM ") and len(text) < 120:
                flush_block()
                current_section = text
                current_subsection = ""
            # Subsection identification based on capitalization
            elif text.isupper() and len(text) < 80:
                flush_block()
                current_subsection = text
            else:
                current_content.append(text)
                
        elif isinstance(child, CT_Tbl):
            tbl = Table(child, doc)
            tbl_md = format_docx_table_to_markdown(tbl)
            if tbl_md:
                current_content.append(tbl_md)
                
    flush_block()
    return blocks

def main():
    print("Starting extraction pipeline...")
    data_dir = get_workspace_paths()
    print(f"Data Directory: {data_dir}")
    
    all_blocks = []
    
    # Part 1: Harvesting Website
    sitemap_path = os.path.join(data_dir, "sitemap.xml")
    urls = parse_sitemap(sitemap_path)
    print(f"Identified {len(urls)} unique, normalized web URLs.")
    
    # Limit to ensure quick extraction if sitemap is large, let's log precisely.
    for index, url in enumerate(urls):
        print(f"[{index + 1}/{len(urls)}] processing...")
        try:
            web_blocks = scrape_web_page(url)
            all_blocks.extend(web_blocks)
        except Exception as error:
            print(f"Scraping error on {url}: {error}")
        # Politeness delay
        time.sleep(0.3)
        
    # Part 2: Processing Franchise Docs (FDD)
    target_docx = None
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.lower().endswith(".docx") and not f.startswith("~$"):
                target_docx = os.path.join(data_dir, f)
                break
                
    if target_docx:
        try:
            docx_blocks = parse_docx(target_docx)
            all_blocks.extend(docx_blocks)
        except Exception as error:
            print(f"Error during document processing: {error}")
    else:
        print("Notice: No DOCX source file identified in data/")
        
    # Part 3: Construct Output Data File
    output_file = os.path.join(data_dir, "extracted_knowledge_base.txt")
    print(f"\nAggregating knowledge into output target: {output_file}")
    
    try:
        with open(output_file, "w", encoding="utf-8") as dest:
            for index, block in enumerate(all_blocks):
                dest.write("---\n")
                dest.write(f"source: {block['source']}\n")
                dest.write(f"url_or_path: {block['url_or_path']}\n")
                dest.write(f"section: {block['section']}\n")
                dest.write(f"subsection: {block['subsection']}\n")
                dest.write("---\n\n")
                dest.write(block["text"])
                dest.write("\n\n\n")
        
        print("Extraction completely successfully!")
        print(f"Count of parsed content blocks: {len(all_blocks)}")
        print(f"Extracted text file available: {output_file}")
    except Exception as e:
        print(f"Failed to save output file: {e}")

if __name__ == "__main__":
    main()
