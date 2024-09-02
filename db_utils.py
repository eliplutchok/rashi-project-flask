from db import get_connection, release_connection

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
    