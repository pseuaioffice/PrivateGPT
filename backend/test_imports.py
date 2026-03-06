try:
    from rag_chain import ask
    print("SUCCESS: rag_chain imports correctly.")
except Exception as e:
    print(f"FAILURE: rag_chain import error: {e}")
