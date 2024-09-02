import openai
import json as JSON
from langchain.embeddings.openai import OpenAIEmbeddings
import asyncio
import httpx
from pydantic import BaseModel, create_model
from typing import Union, Optional
from langsmith.wrappers import wrap_openai
from langsmith.run_helpers import get_current_run_tree
from langsmith import traceable, Client
from prompts import *
from config import *
from pinecone_utils import get_index_endpoint, query_vectors, query_vectors

def feedback_to_langsmith(run_id, score, comment):
    try:
        print(f"Feedback: run_id={run_id}, score={score}, comment={comment}")
        feedback_client = Client()
        feedback_client.create_feedback(
            run_id=run_id,
            key="user_feedback",
            score=int(score),
            comment=comment,
        )
        return "Feedback saved successfully"
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return "Failed to save feedback"