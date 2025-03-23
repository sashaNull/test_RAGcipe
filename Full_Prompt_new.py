import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from openai import OpenAI
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Initialize Cross-Encoder model for reranking
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, metadatas, top_k=3):
    pairs = [(query, doc) for doc in documents]
    scores = cross_encoder_model.predict(pairs)
    ranked_results = sorted(zip(documents, metadatas, scores), key=lambda x: x[2], reverse=True)
    return ranked_results[:top_k]

# ChromaDB setup for recipes (using OpenAI embeddings)
recipes_client = chromadb.PersistentClient(path="chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-ada-002"
)
recipes_collection = recipes_client.get_collection("recipes_collection", embedding_function=openai_ef)

# ChromaDB setup for ingredients (Fairprice with OpenAI embeddings)
ingredients_client = chromadb.PersistentClient(path="fairprice_openai_embeddings_db")
ingredients_collection = ingredients_client.get_collection("fairprice_products_openai", embedding_function=openai_ef)

def generate_prompt(user_query, recipe_name, recipe_url, recipe_details, nutritional_data, ingredients_from_db):
    ingredient_str = ""
    for ing, products in ingredients_from_db.items():
        ingredient_str += f"\n**{ing.capitalize()}** (Price details provided):\n"
        for prod in products:
            meta = prod['metadata']
            product_url = meta.get('url', 'N/A')
            ingredient_str += f"- {meta['name']} by {meta['brand']} (Price: ${meta['price']}, Size: {meta['size']}, URL: {product_url})\n"
    
    prompt = f"""
You are an expert culinary assistant.

A user is seeking recipe suggestions for the query: "**{user_query}**". 
In addition to providing a detailed recipe summary, your task is to help the user make an affordable, healthy purchase by:
1. Analyzing the available FairPrice ingredient options and suggesting suitable ingredient substitutions clearly if any.
2. Identifying the most affordable product for each necessary ingredient.
3. Providing nutritional information clearly based on the provided nutritional data.
4. Optionally estimating the total cost of the required ingredients.

Below is the retrieved recipe and a list of FairPrice ingredient products with their price and source URL information. Please include the source URL for the recipe and each ingredient in your response for clarity and reliability.

---

**Retrieved Recipe:**
- **Recipe Name:** {recipe_name}
- **URL:** {recipe_url}
- **Details:**
{recipe_details}

**Nutritional Information:**
{nutritional_data}

---

**FairPrice Ingredient Products:**
{ingredient_str}

---

Please provide your response in four sections:
1. **Recipe Summary** – Summarize the key steps and ingredients in a concise and clear paragraph, including the recipe source URL.
2. **Affordable Ingredient Recommendations** – For each necessary ingredient, identify the most cost-effective FairPrice product, including its price and source URL.
3. **Nutritional Analysis** – Provide a clear analysis based on the nutritional information of the recipe and the ingredients. Discuss the health benefits or potential dietary advantages (e.g., high protein content, low saturated fat, rich in fiber, etc.). Mention who might benefit from this dish (e.g., vegetarians, fitness enthusiasts, people watching cholesterol).
4. **Cost Estimate** – Provide an estimated total cost for preparing this recipe using the selected FairPrice ingredients.
"""
    return prompt

def get_llm_response(prompt, model="gpt-4o", temperature=0.3):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def extract_ingredients(recipe_text):
    match = re.search(r'Ingredients:(.*?)(Method|Nutritional Info)', recipe_text, re.DOTALL | re.IGNORECASE)
    if match:
        ingredients_block = match.group(1).strip()
        ingredients_list = [
            re.sub(r'[\d\*\(\),]+', '', line).strip().lower()
            for line in ingredients_block.split('\n') if line.strip()
        ]
        return list(set(ingredients_list))
    return []

def search_ingredients_chroma(ingredient_name, top_k=3):
    results = ingredients_collection.query(
        query_texts=[ingredient_name],
        n_results=top_k,
        include=['metadatas', 'documents', 'distances']
    )
    matched_products = []
    for meta, doc, dist in zip(results['metadatas'][0], results['documents'][0], results['distances'][0]):
        matched_products.append({"metadata": meta, "document": doc, "similarity": dist})
    return matched_products

def query_all(query_text, n_results=5):
    print(f"\n🔎 Querying for: {query_text}")

    # Retrieve and rerank recipes
    recipe_results = recipes_collection.query(
        query_texts=[query_text], n_results=n_results, include=['documents', 'metadatas']
    )
    reranked_recipes = rerank(query_text, recipe_results['documents'][0], recipe_results['metadatas'][0], top_k=1)
    recipe_doc, recipe_meta, _ = reranked_recipes[0]

    print("\n🍽️ Top Recipe Selected:")
    print(f"{recipe_meta['name']}")

    # Extract nutritional data if available
    nutritional_data = "Not Available"
    if "Nutritional Info" in recipe_doc:
        nutritional_data = recipe_doc.split("Nutritional Info:")[-1].strip().split("\n\n")[0].strip()

    # Dynamically extract ingredients from recipe text
    ingredients_keywords = extract_ingredients(recipe_doc)

    # Query ingredients dynamically from ChromaDB embeddings
    ingredients_from_db = {
        ing: search_ingredients_chroma(ing) for ing in ingredients_keywords
    }

    # Generate prompt dynamically
    prompt = generate_prompt(
        user_query=query_text,
        recipe_name=recipe_meta['name'],
        recipe_url=recipe_meta['url'],
        recipe_details=recipe_doc,
        nutritional_data=nutritional_data,
        ingredients_from_db=ingredients_from_db
    )

    print("\n📝 Prompt sent to LLM:")
    print(prompt)

    # Generate LLM response
    llm_response = get_llm_response(prompt)
    print("\n✨ LLM-generated Culinary Response:")
    print(llm_response)
    
    # Return the response for evaluation purposes
    return llm_response

if __name__ == "__main__":
    query_text = "cheap high protein tofu dish"
    query_all(query_text)
