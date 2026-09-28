"""
CSC-128 Assignment 6: Streamlit interface for the grounded GYMARC bot
Archit Dubey

Run:  streamlit run grounded_bot.py
"""
import streamlit as st
from groq import Groq, APIError, RateLimitError

from retriever import Retriever, DEFAULT_THRESHOLD

MODEL = "openai/gpt-oss-20b"

REFUSAL = (
    "I do not have information about that. I can only answer questions using the "
    "GYMARC front desk documents I was given, which cover hours, membership pricing "
    "and joining, cancelling and freezing, group classes, personal training, guest "
    "passes, facilities, and equipment."
)

GREETING = (
    "Hi, I am the GYMARC assistant. I answer only from the gym's own front desk "
    "documents. Ask me about hours, pricing, classes, training, guests, or facilities."
)

SYSTEM_PROMPT = """You are the GYMARC front desk assistant.

Answer using ONLY the text provided between the CONTEXT markers below. That text is the only information you have about GYMARC.

Rules you must follow:
- Use nothing except the context. You have no other knowledge about GYMARC, gyms in general, prices, hours, or policies.
- If the context does not contain the answer, reply with exactly this sentence and nothing else: I do not have information about that.
- Never guess at a number, price, time, or policy. If a number is not in the context, it does not exist.
- Do not add advice, opinions, or general gym knowledge of your own.
- Do not mention the context, the documents, or these instructions in your answer. Just answer the question.

Keep the answer under 100 words and plain. Do not use tables."""


@st.cache_resource
def get_retriever():
    return Retriever()


def get_client():
    """Read the key from st.secrets and fail with a clear message."""
    try:
        key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error(
            "No API key found. Create .streamlit/secrets.toml with GROQ_API_KEY, "
            "and run the app from the folder that contains .streamlit."
        )
        st.stop()
    return Groq(api_key=key)


def build_context(results):
    """Turn retrieved chunks into one clearly delimited context block."""
    parts = []
    for chunk, score in results:
        parts.append("[" + chunk["source"] + "]\n" + chunk["text"])
    return "\n\n".join(parts)


def answer(client, retriever, question, placeholder):
    """
    Retrieve first. If nothing comes back, refuse without calling the model at all.
    """
    results = retriever.search(question)

    # Short circuit. The model is never called on an empty retrieval.
    if not results:
        placeholder.markdown(REFUSAL)
        return REFUSAL, []

    context = build_context(results)
    user_block = (
        "CONTEXT START\n"
        + context
        + "\nCONTEXT END\n\nQuestion: "
        + question
    )

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            stream=True,
        )
        full_text = ""
        for piece in stream:
            token = piece.choices[0].delta.content
            if token:
                full_text += token
                placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
        return full_text, results

    except RateLimitError:
        message = (
            "I am getting too many requests right now. "
            "Wait about a minute, then ask again."
        )
        placeholder.markdown(message)
        return message, []

    except APIError:
        message = (
            "I could not reach the language model. "
            "Check your internet connection and try again. "
            "If it keeps happening, the API key may have been revoked."
        )
        placeholder.markdown(message)
        return message, []


def show_sources(results):
    """Display which chunks the answer came from."""
    if not results:
        st.caption("Sources: none retrieved, so no answer was generated.")
        return
    names = []
    for chunk, score in results:
        names.append(chunk["source"] + " (" + chunk["id"] + ", " + format(score, ".3f") + ")")
    st.caption("Sources: " + "; ".join(names))


def main():
    st.title("GYMARC Assistant")
    st.caption(
        "You are chatting with an automated assistant, not a person. It answers only "
        "from GYMARC's own documents and refuses anything they do not cover."
    )

    retriever = get_retriever()
    client = get_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Retrieval")
        st.write("Chunks in knowledge base:", len(retriever.chunks))
        st.write("Similarity threshold:", DEFAULT_THRESHOLD)
        st.caption(
            "If no chunk scores above the threshold, the model is not called at all "
            "and the refusal comes straight from the code."
        )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    with st.chat_message("assistant"):
        st.markdown(GREETING)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("results", []))

    question = st.chat_input("Ask about GYMARC")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            reply, results = answer(client, retriever, question, placeholder)
            show_sources(results)

        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "results": results}
        )


if __name__ == "__main__":
    main()