import re
from typing import Optional

def clean_and_normalize_text(text: str, preserve_case: bool = True, normalize_unicode: bool = True) -> str:
    """
    Cleans and normalizes text for embedding use.
    
    Args:
        text (str): Raw input text (e.g., from PDF).
        preserve_case (bool): If False, lowercases the text (default: True).
        normalize_unicode (bool): If True, normalize unicode dashes, quotes, etc. (default: True).
    
    Returns:
        str: Cleaned, normalized text ready for chunking/embedding.
    """

    # 1. Unicode normalization (smart quotes, dashes, etc.)
    if normalize_unicode:
        text = text.replace("\u201c", '"').replace("\u201d", '"')  # smart quotes
        text = text.replace("\u2018", "'").replace("\u2019", "'")  # smart apostrophes
        text = text.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash
        text = text.replace("\xa0", " ")  # non-breaking space

    # 2. Fix line breaks within paragraphs (i.e., mid-sentence)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 3. Normalize multiple newlines to paragraph breaks (i.e., section separation)
    text = re.sub(r'\n{2,}', '\n\n', text)

    # 4. Normalize excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)

    # 5. Strip leading/trailing whitespace
    text = text.strip()

    # 6. Optional lowercase
    if not preserve_case:
        text = text.lower()

    return text
