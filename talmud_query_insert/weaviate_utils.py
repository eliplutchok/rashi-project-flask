import requests
import json
from langsmith import traceable
from talmud_query.config import WEAVIATE_API_KEY, WEAVIATE_URL
from talmud_query.embed_utils import embed_text_openai

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {WEAVIATE_API_KEY}"
}

@traceable
def get_weaviate_vdb_results(embedded_query, weaviate_url, limit=10, filter_obj=None):
    query = """
    query($vector: [Float!]!, $limit: Int!, $where: QuestionWhereInput) {
      Get {
        Question (
          limit: $limit
          nearVector: {
            vector: $vector
          }
          where: $where
        ) {
          english_text
          hebrew_text
          book_name
          page_number
          passage_id
          translation_id
        }
      }
    }
    """

    variables = {
        "vector": embedded_query,
        "limit": limit,
        "where": filter_obj
    }

    response = requests.post(
        f"{weaviate_url}/v1/graphql",
        headers=headers,
        json={"query": query, "variables": variables}
    )
    response.raise_for_status()
    results = response.json()

    passages = [
        {
            'passage_id': result['passage_id'],
            'hebrew_text': result['hebrew_text'],
            'english_text': result['english_text'],
            'translation_id': result['translation_id'],
            'book_name': result['book_name'],
            'page_number': result['page_number'],
            'text_to_embed': result['text_to_embed']
        }
        for result in results['data']['Get']['Question']
    ]

    # Filter out passages that have English text which includes "sample translation"
    passages = [passage for passage in passages if "sample translation" not in passage['english_text'].lower()]
    return passages

@traceable
def get_context_from_weaviate_vdb(queries, weaviate_url, k=10, filter_obj=None, print_output=False):
    contexts = []

    for key in queries:
        if key.startswith("query"):
            embedded_query = embed_text_openai(queries[key])[0]  # Get the first (and only) embedding
            context = get_weaviate_vdb_results(embedded_query, weaviate_url, limit=k, filter_obj=filter_obj)
            contexts.extend(context)

    # Remove duplicates
    contexts = [dict(t) for t in {tuple(d.items()) for d in contexts}]
    
    if print_output:
        print("Number of contexts: ", len(contexts))
    return contexts

# Example usage:
if __name__ == "__main__":
    search_concept = "biology"
    weaviate_url = "https://your-weaviate-instance-url.com"  # Replace with actual URL
    filter_obj = {
        "operator": "And",
        "operands": [
            {
                "path": ["book_name"],
                "operator": "Equal",
                "valueString": "Specific Book"
            }
        ]
    }
    results = get_context_from_weaviate_vdb(
        {"query1": search_concept},
        weaviate_url,
        k=2,
        filter_obj=filter_obj,
        print_output=True
    )
    print(json.dumps(results, indent=2))