from Full_Prompt_new import query_all

# Define a set of test queries
test_queries = [
    "high protein tofu dish",
    "low carb vegetarian meal",
    "halal tom yam soup under $3",
    "quick chicken stir fry"
]

def evaluate_queries():
    for query in test_queries:
        print(f"\n=== Evaluating Query: {query} ===")
        response = query_all(query)
        print("\n--- LLM Response ---\n", response)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    evaluate_queries()
