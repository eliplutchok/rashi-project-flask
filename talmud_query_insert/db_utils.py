from db import get_connection, release_connection

import re

def break_into_sentences(text):
    # Regular expression to match any punctuation mark followed by a space
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def get_only_bolded_words(text):
    # Regular expression to match any words surrounded by bold tags
    bolded_words = re.findall(r'<b>(.*?)</b>', text)
    # return as one string
    return ' '.join(bolded_words)   

# Step 1: Fetch all passages from the database for books that do NOT include "rashi" in their name
def fetch_passages():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Perform a join between passages and books, filtering for books that do not have "rashi" in their name
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text 
                FROM passages
                JOIN books ON passages.book_id = books.book_id
                WHERE books.name NOT ILIKE '%rashi%'
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    return passages

def fetch_english_passages():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Perform a join between passages and books, filtering for books that do not have "rashi" in their name
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id
                JOIN books ON passages.book_id = books.book_id
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE books.name NOT ILIKE '%rashi%'
                AND translations.version_name = 'Sefaria-William-Davidson'
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    formatted_passages = []
    for passage in passages:
        formatted_passages.append({
            'text_to_embed': passage[2],
            'passage_id': passage[0],
            'hebrew_text': passage[1],
            'english_text': passage[2],
            'translation_id': passage[3],
            'book_name': passage[4],
            'page_number': passage[5]
        })
    
    return formatted_passages

def fetch_hebrew_passages():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Perform a join between passages and books, filtering for books that do not have "rashi" in their name
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id
                JOIN books ON passages.book_id = books.book_id
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE books.name NOT ILIKE '%rashi%'
                AND translations.version_name = 'Sefaria-William-Davidson'
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    formatted_passages = []
    for passage in passages:
        formatted_passages.append({
            'text_to_embed': passage[1],
            'passage_id': passage[0],
            'hebrew_text': passage[1],
            'english_text': passage[2],
            'translation_id': passage[3],
            'book_name': passage[4],
            'page_number': passage[5]
        })
    
    return formatted_passages

def fetch_english_pages():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number, passages.passage_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id
                JOIN books ON passages.book_id = books.book_id
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE books.name NOT ILIKE '%rashi%'
                AND translations.version_name = 'Sefaria-William-Davidson'
                ORDER BY books.name, pages.page_number, passages.passage_number
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    formatted_pages = {}
    for passage in passages:
        page_key = (passage[4], passage[5])  # (book_name, page_number)
        if page_key not in formatted_pages:
            formatted_pages[page_key] = {
                'text_to_embed': '',
                'hebrew_text': '',
                'english_text': '',
                'passage_ids': [],
                'translation_ids': [],
                'book_name': passage[4],
                'page_number': passage[5]
            }
        
        # Ensure no duplicates
        if passage[0] not in formatted_pages[page_key]['passage_ids']:
            formatted_pages[page_key]['text_to_embed'] += ' ' + passage[2]
            formatted_pages[page_key]['hebrew_text'] += ' ' + passage[1]
            formatted_pages[page_key]['english_text'] += ' ' + passage[2]
            formatted_pages[page_key]['passage_ids'].append(passage[0])
            formatted_pages[page_key]['translation_ids'].append(passage[3])
    
    return list(formatted_pages.values())

def fetch_hebrew_pages():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number, passages.passage_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id
                JOIN books ON passages.book_id = books.book_id
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE books.name NOT ILIKE '%rashi%'
                AND translations.version_name = 'Sefaria-William-Davidson'
                ORDER BY books.name, pages.page_number, passages.passage_number
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    formatted_pages = {}
    for passage in passages:
        page_key = (passage[4], passage[5])  # (book_name, page_number)
        if page_key not in formatted_pages:
            formatted_pages[page_key] = {
                'text_to_embed': '',
                'hebrew_text': '',
                'english_text': '',
                'passage_ids': [],
                'translation_ids': [],
                'book_name': passage[4],
                'page_number': passage[5]
            }
        
        if passage[0] not in formatted_pages[page_key]['passage_ids']:
            formatted_pages[page_key]['text_to_embed'] += ' ' + passage[1]
            formatted_pages[page_key]['hebrew_text'] += ' ' + passage[1]
            formatted_pages[page_key]['english_text'] += ' ' + passage[2]
            formatted_pages[page_key]['passage_ids'].append(passage[0])
            formatted_pages[page_key]['translation_ids'].append(passage[3])
    
    return list(formatted_pages.values())

def fetch_english_pages_bolded():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number, passages.passage_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id
                JOIN books ON passages.book_id = books.book_id
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE books.name NOT ILIKE '%rashi%'
                AND translations.version_name = 'Sefaria-William-Davidson'
                ORDER BY books.name, pages.page_number, passages.passage_number
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    formatted_pages = {}
    for passage in passages:
        page_key = (passage[4], passage[5])  # (book_name, page_number)
        if page_key not in formatted_pages:
            formatted_pages[page_key] = {
                'text_to_embed': '',
                'english_text': '',
                'hebrew_text': '',
                'passage_ids': [],
                'translation_ids': [],
                'book_name': passage[4],
                'page_number': passage[5]
            }
        
        if passage[0] not in formatted_pages[page_key]['passage_ids']:
            bolded_text = get_only_bolded_words(passage[2])
            formatted_pages[page_key]['text_to_embed'] += ' ' + bolded_text
            formatted_pages[page_key]['english_text'] += ' ' + bolded_text
            formatted_pages[page_key]['hebrew_text'] += ' ' + passage[1]
            formatted_pages[page_key]['passage_ids'].append(passage[0])
            formatted_pages[page_key]['translation_ids'].append(passage[3])
    
    return list(formatted_pages.values())

def fetch_grouped_passages(language, group_size):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number, passages.passage_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id
                JOIN books ON passages.book_id = books.book_id
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE books.name NOT ILIKE '%rashi%'
                AND translations.version_name = 'Sefaria-William-Davidson'
                ORDER BY books.name, pages.page_number, passages.passage_number
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    formatted_groups = []
    current_group = None
    current_count = 0
    current_book = None

    for passage in passages:
        if current_book != passage[4] or current_count >= group_size:
            if current_group:
                formatted_groups.append(current_group)
            current_group = {
                'text_to_embed': '',
                'hebrew_text': '',
                'english_text': '',
                'passage_ids': [],
                'translation_ids': [],
                'book_name': passage[4],
                'start_page': passage[5],
                'end_page': passage[5]
            }
            current_count = 0
            current_book = passage[4]

        text_to_embed = passage[1] if language == 'hebrew' else get_only_bolded_words(passage[2])
        current_group['text_to_embed'] += ' ' + text_to_embed
        current_group['hebrew_text'] += ' ' + passage[1]
        current_group['english_text'] += ' ' + passage[2]
        current_group['passage_ids'].append(passage[0])
        current_group['translation_ids'].append(passage[3])
        current_group['end_page'] = passage[5]
        current_count += 1

    if current_group:
        formatted_groups.append(current_group)

    return formatted_groups

def fetch_sentence_passages():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Perform a join between passages and books, filtering for books that do not have "rashi" in their name
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id
                JOIN books ON passages.book_id = books.book_id
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE books.name NOT ILIKE '%rashi%'
                AND translations.version_name = 'Sefaria-William-Davidson'
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    formatted_passages = []
    for passage in passages:
        for sentence in break_into_sentences(passage[2]):
            formatted_passages.append({
                'text_to_embed': sentence,
                'passage_id': passage[0],
                'hebrew_text': passage[1],
                'english_text': passage[2],
                'translation_id': passage[3],
                'book_name': passage[4],
                'page_number': passage[5]
            })

    return formatted_passages

def fetch_bolded_words_passages():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Perform a join between passages and books, filtering for books that do not have "rashi" in their name
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id
                JOIN books ON passages.book_id = books.book_id
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE books.name NOT ILIKE '%rashi%'
                AND translations.version_name = 'Sefaria-William-Davidson'
            """)
            passages = cursor.fetchall()
    finally:
        release_connection(conn)
    
    formatted_passages = []
    for passage in passages:
        formatted_passages.append({
            'text_to_embed': get_only_bolded_words(passage[2]),
            'passage_id': passage[0],
            'hebrew_text': passage[1],
            'english_text': passage[2],
            'translation_id': passage[3],
            'book_name': passage[4],
            'page_number': passage[5]
        })

    return formatted_passages

def get_passage_text(passage_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT hebrew_text
                FROM passages
                WHERE passage_id = %s
            """, (passage_id,))
            passage = cursor.fetchone()
    finally:
        release_connection(conn)
    
    return passage[0]

def get_translation_text(translation_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT text
                FROM translations
                WHERE translation_id = %s
            """, (translation_id,))
            translation = cursor.fetchone()
    finally:
        release_connection(conn)
    
    return translation[0]


def get_passage_and_translation(passage_id, version_name):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT passages.passage_id, passages.hebrew_text, translations.text, translations.translation_id, books.name, pages.page_number
                FROM passages
                JOIN pages ON passages.page_id = pages.page_id      
                JOIN translations ON passages.passage_id = translations.passage_id
                WHERE passages.passage_id = %s
                AND translations.version_name = %s
            """, (passage_id, version_name))
            passage = cursor.fetchone()
    finally:
        release_connection(conn)
    
    formatted_passage = {
        'passage_id': passage[0],
        'hebrew_text': passage[1],
        'english_text': passage[2],
        'translation_id': passage[3],
        'book_name': passage[4],
        'page_number': passage[5]
    }

    return formatted_passage


def get_nearby_passages(passage_id, num_passages):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # First, get the book_id and passage_number of the input passage
            cursor.execute("""
                SELECT book_id, passage_number
                FROM passages
                WHERE passage_id = %s
            """, (passage_id,))
            current_passage = cursor.fetchone()
            if not current_passage:
                return []

            book_id, passage_number = current_passage

            # Then, fetch nearby passages
            cursor.execute("""
                SELECT p.passage_id, p.hebrew_text, t.text, t.translation_id, b.name, pg.page_number
                FROM passages p
                JOIN books b ON p.book_id = b.book_id
                JOIN pages pg ON p.page_id = pg.page_id
                JOIN translations t ON p.passage_id = t.passage_id
                WHERE p.book_id = %s
                AND p.passage_number BETWEEN %s AND %s
                AND t.version_name = 'Sefaria-William-Davidson'
                ORDER BY p.passage_number
                LIMIT %s
            """, (book_id, passage_number - num_passages, passage_number + num_passages, 2 * num_passages + 1))
            
            passages = cursor.fetchall()

    finally:
        release_connection(conn)
    
    formatted_passages = []
    for passage in passages:
        formatted_passages.append({
            'passage_id': passage[0],
            'hebrew_text': passage[1],
            'english_text': passage[2],
            'translation_id': passage[3],
            'book_name': passage[4],
            'page_number': passage[5]
        })
    
    return formatted_passages

 
    