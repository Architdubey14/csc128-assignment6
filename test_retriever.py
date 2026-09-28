"""
CSC-128 Assignment 6: retrieval tests for the GYMARC bot
Archit Dubey

Run:  python test_retriever.py

No API key is needed. Retrieval is deterministic, so this runs on its own.
"""
from retriever import Retriever, DEFAULT_THRESHOLD

# Questions a member would really ask, with the chunk id that should come back.
SHOULD_RETRIEVE = [
    ("what time do you open on saturday", "hours"),
    ("how much is a membership", "membership_price"),
    ("is there a student discount", "membership_price"),
    ("i want to cancel my membership", "cancel"),
    ("i want a break from the gym for a couple months", "freeze"),
    ("do you have yoga classes", "classes"),
    ("how much does a trainer cost", "personal_training"),
    ("can i bring a friend with me", "guests"),
    ("do you have a sauna", "facilities"),
    ("do i need to bring my own towel", "facilities"),
    ("do you have squat racks", "equipment"),
    ("how do i sign up", "joining"),
    ("is parking free", "facilities"),
]

# Close to a gym, but not covered by any chunk. These must retrieve nothing.
SHOULD_REFUSE = [
    "do you sell protein powder",
    "are you hiring right now",
    "can i get a massage here",
    "is there a basketball court",
    "do you have a tanning bed",
]


def run_tests():
    retriever = Retriever()
    passed = 0
    failed = 0

    retrieve_scores = []
    refuse_scores = []

    print("SHOULD RETRIEVE")
    print("-" * 70)
    for question, expected_id in SHOULD_RETRIEVE:
        raw = retriever.best_score(question)
        retrieve_scores.append(raw)

        results = retriever.search(question)
        actual_id = None
        if results:
            actual_id = results[0][0]["id"]

        if actual_id == expected_id:
            passed += 1
            mark = "ok  "
        else:
            failed += 1
            mark = "FAIL"

        print(f"{mark}  {raw:.3f}  {question}")
        if actual_id != expected_id:
            print(f"        expected {expected_id}, got {actual_id}")

    print()
    print("SHOULD REFUSE")
    print("-" * 70)
    for question in SHOULD_REFUSE:
        raw = retriever.best_score(question)
        refuse_scores.append(raw)

        results = retriever.search(question)
        if not results:
            passed += 1
            mark = "ok  "
        else:
            failed += 1
            mark = "FAIL"

        print(f"{mark}  {raw:.3f}  {question}")
        if results:
            print(f"        retrieved {results[0][0]['id']} but should have refused")

    print()
    print("-" * 70)
    print("Threshold in use:", DEFAULT_THRESHOLD)
    print("Lowest score among should-retrieve:", round(min(retrieve_scores), 3))
    print("Highest score among should-refuse: ", round(max(refuse_scores), 3))
    print("Gap:", round(min(retrieve_scores) - max(refuse_scores), 3))
    print()
    print("Passed:", passed)
    print("Failed:", failed)
    print("Total: ", passed + failed)


if __name__ == "__main__":
    run_tests()