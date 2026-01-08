from rag.pipeline import answer

TESTS = [
    # In-scope (should answer + cite)
    ("What is self-attention and why is it useful?", "answer_with_citations"),
    ("What are key components of the Transformer architecture?", "answer_with_citations"),
    ("What problem does the Transformer address compared to recurrent models?", "answer_with_citations"),

    # EU AI Act (should answer + cite)
    ("What are the main themes or obligations discussed in the EU AI Act document?", "answer_with_citations"),

    # Out-of-scope (should refuse)
    ("Who won the 2022 World Cup?", "refuse"),
]

REFUSAL = "I don't know based on the provided documents."

def has_citation(text: str) -> bool:
    return "[" in text and "]" in text  

def main():
    print("\n=== CeADAR RAG Evaluation (Lightweight) ===\n")
    total = len(TESTS)
    pass_count = 0

    for q, expected in TESTS:
        res = answer(q, top_k=4)
        out = (res["answer"] or "").strip()

        ok = False
        if expected == "refuse":
            ok = (out == REFUSAL)
        else:
            ok = (out != REFUSAL) and has_citation(out)

        status = "✅ PASS" if ok else "❌ FAIL"
        print("-" * 90)
        print(f"{status} | Expected: {expected}")
        print("Q:", q)
        print("A:", out[:600] + ("..." if len(out) > 600 else ""))
        print("Top sources:", [f"{s['source_file']} p={s['page']}" for s in res["sources"][:2]])

        if ok:
            pass_count += 1

    print("\n" + "=" * 90)
    print(f"Score: {pass_count}/{total} passed")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
