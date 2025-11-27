"""
Final Resume Parser Module
- Works with PDFs
- Extracts Name, Email, Phone, Skills, Education
- Provides section splitting
- Provides comprehensive parsing function for app.py
- Fixes spaced-out characters (H A R S H -> Harsh)
"""

import re
from typing import List, Dict, Optional

# Multiple PDF extraction methods with fallbacks
try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ✅ Load spaCy for NER
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (OSError, ImportError):
    # Fallback if spaCy or model not installed
    nlp = None

# ✅ Predefined skill list (extendable)
COMMON_SKILLS = [
    "Python", "Java", "C++", "C", "SQL", "HTML", "CSS", "JavaScript",
    "React", "Node.js", "Django", "Flask", "TensorFlow", "PyTorch",
    "Machine Learning", "Deep Learning", "NLP", "Data Science",
    "AWS", "Docker", "Kubernetes", "Git", "Excel", "Power BI", "Tableau",
    "PHP", "Ruby", "Go", "R", "Matlab", "Hadoop", "Spark"
]

# -------------------------
# Text Normalization
# -------------------------

def normalize_text(text: str) -> str:
    """
    Fix text where characters are spaced out like:
    'H A R S H  P A N C H A L' -> 'Harsh Panchal'
    """
    lines = text.splitlines()
    normalized_lines = []

    for line in lines:
        # Split words using 2 or more spaces as separator between words
        parts = re.split(r'\s{2,}', line)  # 2 or more spaces separate words
        new_parts = []
        for part in parts:
            letters = part.split()  # split single letters
            # If most are single letters, join them
            if len(letters) > 1 and all(len(l) == 1 for l in letters):
                new_parts.append(''.join(letters).title())
            else:
                new_parts.append(part)
        normalized_lines.append(' '.join(new_parts))  # join words with single space

    return '\n'.join(normalized_lines)


# -------------------------
# Basic Extractors
# -------------------------

def extract_text_pdf(path: str) -> str:
    """
    Extract raw text from a PDF file using multiple fallback methods.
    Tries: PyMuPDF (fastest), pdfplumber (best for tables), pdfminer (fallback)
    """
    text = ""
    errors = []
    
    # Method 1: PyMuPDF (fastest, handles most formats)
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            text = "\n".join(text_parts)
            if text and len(text.strip()) > 10:  # Valid text extracted
                return normalize_text(text)
        except Exception as e:
            errors.append(f"PyMuPDF: {str(e)}")
    
    # Method 2: pdfplumber (excellent for complex layouts)
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            text = "\n".join(text_parts)
            if text and len(text.strip()) > 10:
                return normalize_text(text)
        except Exception as e:
            errors.append(f"pdfplumber: {str(e)}")
    
    # Method 3: pdfminer (old reliable fallback)
    if PDFMINER_AVAILABLE:
        try:
            text = pdfminer_extract(path)
            if text and len(text.strip()) > 10:
                return normalize_text(text)
        except Exception as e:
            errors.append(f"pdfminer: {str(e)}")
    
    # If all methods failed, return error message
    if not text or len(text.strip()) < 10:
        raise ValueError(f"Failed to extract text from PDF. Errors: {'; '.join(errors) if errors else 'No PDF libraries available'}")
    
    return normalize_text(text)


def extract_email(text: str) -> Optional[str]:
    """Extract the first email address found"""
    matches = re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)
    return matches[0] if matches else None


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number (handles continuous, dashed, or spaced numbers)."""
    phone_pattern = re.compile(
        r'(\+?\d{1,3}[\s\-]?)?(\d{10}|\d{3}[\s\-]\d{3}[\s\-]\d{4}|\d{5}[\s\-]\d{5})'
    )
    match = phone_pattern.search(text)
    if match:
        number = match.group(0)
        # Clean number (remove spaces and dashes)
        cleaned = re.sub(r"[^\d+]", "", number)
        return cleaned
    return None


def extract_name(text: str) -> Optional[str]:
    """
    Extract candidate name from the resume TEXT content (not filename).
    Name is typically at the very top of the resume (first line).
    """
    
    # Split into lines and get the first non-empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    if not lines:
        return None
    
    # Common words/phrases that are NOT names (to skip)
    skip_patterns = [
        'resume', 'curriculum', 'vitae', 'cv', 'portfolio',
        'objective', 'summary', 'profile', 'about',
        'education', 'experience', 'skills', 'projects',
        'email', 'phone', 'address', 'contact', 'linkedin',
        'www', 'http', 'the', 'a', 'an'
    ]
    
    # Try spaCy NER first on the first few lines
    if nlp is not None and len(lines) > 0:
        try:
            # Analyze first 2 lines (name is usually there)
            preview = '\n'.join(lines[:2])[:300]
            doc = nlp(preview)
            
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    words = name.split()
                    # Validate: 2-4 words for a proper name
                    if 2 <= len(words) <= 4:
                        first_word = words[0].lower().strip('.,;:!?')
                        # Check it's not a skip word
                        if first_word not in skip_patterns:
                            # Ensure it's a valid name (not all caps header)
                            if not name.isupper() or len(name) < 15:
                                return name.title()
        except Exception:
            pass
    
    # Focus on FIRST LINE ONLY - name is almost always the first line
    if len(lines) > 0:
        first_line = lines[0].strip()
        first_line_lower = first_line.lower()
        
        # Skip obvious non-name patterns
        if any(first_line_lower.startswith(pattern) for pattern in skip_patterns):
            pass  # Move to second line
        elif re.match(r'^[@\+]|\d{10,}|http|www\.', first_line, re.I):
            pass  # Skip lines starting with contact info
        elif re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', first_line):
            pass  # Skip dates
        elif first_line.isupper() and len(first_line) > 20:
            pass  # Skip all-caps headers
        else:
            # Extract words from first line
            words = [w.strip('.,;:!?()[]') for w in first_line.split() if w.strip('.,;:!?()[]')]
            
            # Name should have 2-4 words
            if 2 <= len(words) <= 4:
                # Check if first word starts with capital letter
                if words[0] and words[0][0].isupper() and words[0][0].isalpha():
                    first_word_lower = words[0].lower()
                    if first_word_lower not in skip_patterns:
                        # Count how many words are properly capitalized
                        capital_count = sum(1 for w in words if w and w[0].isupper() and w[0].isalpha())
                        # At least first two words should be capitalized (Firstname Lastname)
                        if capital_count >= min(2, len(words)):
                            # Return the full name from first line
                            full_name = ' '.join(words)
                            # Only return if it looks like a name (not all caps, reasonable length)
                            if not full_name.isupper() or len(full_name) < 20:
                                return full_name.title()
    
    # Try second line if first line didn't work
    if len(lines) > 1:
        second_line = lines[1].strip()
        second_line_lower = second_line.lower()
        
        if not any(second_line_lower.startswith(pattern) for pattern in skip_patterns):
            if not re.match(r'^[@\+]|\d{10,}|http|www\.', second_line, re.I):
                words = [w.strip('.,;:!?()[]') for w in second_line.split() if w.strip('.,;:!?()[]')]
                if 2 <= len(words) <= 4:
                    if words[0] and words[0][0].isupper() and words[0][0].isalpha():
                        first_word_lower = words[0].lower()
                        if first_word_lower not in skip_patterns:
                            capital_count = sum(1 for w in words if w and w[0].isupper() and w[0].isalpha())
                            if capital_count >= min(2, len(words)):
                                full_name = ' '.join(words)
                                if not full_name.isupper() or len(full_name) < 20:
                                    return full_name.title()
    
    # If still nothing found, return None (don't use filename - let it be null)
    return None

# -------------------------
# Section Splitter
# -------------------------

def simple_section_split(text: str) -> Dict[str, str]:
    """
    Simple resume section splitter based on keywords
    """
    sections = {}
    current_section = "general"
    sections[current_section] = []

    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue

        if re.search(r"(education|qualification)", line_clean, re.I):
            current_section = "education"
            sections[current_section] = []
        elif re.search(r"(experience|employment|work history)", line_clean, re.I):
            current_section = "experience"
            sections[current_section] = []
        elif re.search(r"(skill|technology|tools)", line_clean, re.I):
            current_section = "skills"
            sections[current_section] = []
        elif re.search(r"(project|publication)", line_clean, re.I):
            current_section = "projects"
            sections[current_section] = []

        sections[current_section].append(line_clean)

    return {k: "\n".join(v) for k, v in sections.items()}

# -------------------------
# Improved Extractors
# -------------------------

def extract_skills(text: str, custom_skill_list: List[str] = None) -> List[str]:
    """Extract skills using keyword matching"""
    skills_found = set()
    skill_list = COMMON_SKILLS + (custom_skill_list or [])

    for skill in skill_list:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.I):
            skills_found.add(skill)

    return list(skills_found)


def extract_education(text: str) -> List[str]:
    """Extract education qualifications"""
    edu_keywords = [
        "B.Sc", "M.Sc", "B.Tech", "M.Tech", "MBA", "PhD",
        "Bachelor", "Master", "Diploma", "BE", "ME", "BCA", "MCA"
    ]
    found = []
    for kw in edu_keywords:
        if re.search(kw, text, re.I):
            found.append(kw)
    return list(set(found))

# -------------------------
# Main Parser
# -------------------------

def parse_resume_comprehensive(path: str) -> Dict[str, any]:
    """
    Comprehensive resume parsing that extracts all relevant information
    Required by app.py
    """
    try:
        text = extract_text_pdf(path)
        if not text or len(text.strip()) < 20:
            raise ValueError("Extracted text is too short or empty")
    except Exception as e:
        # Return minimal structure with error info
        return {
            "text": "",
            "sections": {},
            "skills": [],
            "education": [],
            "email": None,
            "phone": None,
            "name": None,
            "error": str(e)
        }
    
    try:
        sections = simple_section_split(text)
    except Exception:
        sections = {"general": text}
    
    # Extract fields with individual error handling
    try:
        skills = extract_skills(text)
    except Exception:
        skills = []
    
    try:
        education = extract_education(text)
    except Exception:
        education = []
    
    try:
        email = extract_email(text)
    except Exception:
        email = None
    
    try:
        phone = extract_phone(text)
    except Exception:
        phone = None
    
    try:
        name = extract_name(text)
    except Exception:
        name = None

    return {
        "text": text,
        "sections": sections,
        "skills": skills,
        "education": education,
        "email": email,
        "phone": phone,
        "name": name
    }
