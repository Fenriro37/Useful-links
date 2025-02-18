import openai
import numpy as np

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    # The result is a dictionary; 'data'[0]' contains the embedding
    return response["data"][0]["embedding"]

# Example input: a list of chunks and candidate questions for each chunk
chunks = [
    "Chunk 1 text...",
    "Chunk 2 text...",
    # ...
]

# Suppose for each chunk you already have a list of candidate questions
candidate_questions_for_chunk = {
    0: ["Question 1 for chunk 1", "Question 2 for chunk 1", ...],
    1: ["Question 1 for chunk 2", "Question 2 for chunk 2", ...],
    # ...
}

chosen_questions = []
chosen_questions_embeddings = []

SIMILARITY_THRESHOLD = 0.80  # for diversity

for i, chunk in enumerate(chunks):
    # 1) Get embedding for the chunk
    chunk_embedding = get_embedding(chunk)

    # 2) Get embeddings for all candidate questions
    question_embeddings = []
    for q in candidate_questions_for_chunk[i]:
        q_embedding = get_embedding(q)
        question_embeddings.append((q, q_embedding))

    # 3) Rank by similarity to the chunk
    #    The higher the cosine similarity, the more relevant the question
    ranked_questions = []
    for q, q_emb in question_embeddings:
        sim = cosine_similarity(q_emb, chunk_embedding)
        ranked_questions.append((q, q_emb, sim))

    # Sort descending by similarity
    ranked_questions.sort(key=lambda x: x[2], reverse=True)

    # 4) Enforce diversity
    best_question = None
    best_question_embedding = None

    for (q, q_emb, sim) in ranked_questions:
        # Check if this question is too similar to previously chosen questions
        too_similar = False
        for chosen_emb in chosen_questions_embeddings:
            if cosine_similarity(q_emb, chosen_emb) > SIMILARITY_THRESHOLD:
                too_similar = True
                break

        if not too_similar:
            best_question = q
            best_question_embedding = q_emb
            break

    # If all were too similar, you could relax your threshold or pick the top anyway
    if not best_question:
        # fallback to picking the top question ignoring diversity
        best_question, best_question_embedding, _ = ranked_questions[0]

    chosen_questions.append(best_question)
    chosen_questions_embeddings.append(best_question_embedding)

# At the end, chosen_questions[i] contains the best question for chunk i.


---------------------------------------------------------------------------------------------

def generate_candidate_questions(chunk_text, n=5):
    """
    Use GPT to generate candidate questions, but do not ask it to rate or score them.
    """
    # Example prompt (minimal):
    prompt = f"""
    Based on the following text, please generate {n} different questions:
    {chunk_text}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    # Parse out the questions from 'response'
    candidate_questions = parse_questions(response)
    return candidate_questions

import openai
import numpy as np

def get_embedding(text):
    """
    Compute the embedding of the input text/string using a chosen model.
    """
    embed_response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    embedding = embed_response['data'][0]['embedding']
    return np.array(embedding)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def pick_best_question_by_embedding(chunk_text, candidate_questions):
    chunk_embedding = get_embedding(chunk_text)
    # Score each question by similarity to the chunk
    question_scores = []
    for q in candidate_questions:
        q_embedding = get_embedding(q)
        sim = cosine_similarity(chunk_embedding, q_embedding)
        question_scores.append((q, q_embedding, sim))
    
    # Sort by similarity descending
    question_scores.sort(key=lambda x: x[2], reverse=True)
    
    # The top candidate is the best match to the chunk
    best_question, best_embedding, best_sim = question_scores[0]
    return best_question, best_embedding

def pick_diverse_question_by_embedding(chunk_text, candidate_questions, chosen_questions_embeddings, 
                                       chunk_similarity_threshold=0.75,
                                       inter_question_diversity_threshold=0.80):
    """
    chunk_similarity_threshold    => not typically used if all questions are relevant, 
                                     but could skip if below some min threshold.
    inter_question_diversity_threshold => skip questions that are too similar to previously chosen ones.
    """
    chunk_embedding = get_embedding(chunk_text)
    question_scores = []
    
    for q in candidate_questions:
        q_embedding = get_embedding(q)
        sim = cosine_similarity(chunk_embedding, q_embedding)
        question_scores.append((q, q_embedding, sim))
    
    # Sort by how relevant the question is to the chunk
    question_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Iterate from best match downward, checking for diversity
    for q, q_embed, sim in question_scores:
        # (Optional) skip if not relevant enough
        if sim < chunk_similarity_threshold:
            continue
        
        # Check for diversity vs. previously chosen questions
        too_similar = False
        for chosen_q_embed in chosen_questions_embeddings:
            similarity_to_chosen = cosine_similarity(q_embed, chosen_q_embed)
            if similarity_to_chosen > inter_question_diversity_threshold:
                too_similar = True
                break
        
        # If it's sufficiently different, pick it
        if not too_similar:
            return q, q_embed
    
    # Fallback if all are "too similar" => pick the top anyway
    return question_scores[0][0], question_scores[0][1]
chunks = [...]  # Your text chunks
chosen_questions = []
chosen_questions_embeddings = []

for chunk in chunks:
    # 1. Generate candidate questions (e.g., 5 each)
    candidate_questions = generate_candidate_questions(chunk, n=5)

    # 2. Pick the best question using only embeddings (and ensuring diversity)
    best_question, best_embedding = pick_diverse_question_by_embedding(
        chunk_text=chunk,
        candidate_questions=candidate_questions,
        chosen_questions_embeddings=chosen_questions_embeddings,
        chunk_similarity_threshold=0.75,
        inter_question_diversity_threshold=0.80
    )

    # 3. Record final selection
    chosen_questions.append(best_question)
    chosen_questions_embeddings.append(best_embedding)

# 'chosen_questions' now contains the single best (and diverse) question per chunk.

