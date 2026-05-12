import docx
def test():
    doc = docx.Document('/Users/aryamantyagi/Downloads/2026 WIN FDD(82160923.5).docx')
    sections = []
    current_section = "General FDD Info"
    current_text = []
    
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        if text.startswith("ITEM ") and "\t" not in text:
            if current_text:
                sections.append((current_section, "\n\n".join(current_text)))
            current_section = text
            current_text = []
        else:
            current_text.append(text)
            
    if current_text:
        sections.append((current_section, "\n\n".join(current_text)))
        
    for s in sections[:3]:
        print(s[0])
        print(s[1][:100] + "...")
test()
