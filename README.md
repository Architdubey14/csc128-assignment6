# GYMARC Grounded Bot

CSC-128 Assignment 6

Archit Dubey

## Overview

A Streamlit bot answering questions about GYMARC with only a knowledge base I wrote. The user asks something and a TF-IDF retriever finds the documents to include, then those documents are the only things passed to the language model. If nothing is retrieved then the model is never called and the bot refuses.

This is the solution to the problem in Assignment 5, where my bot invented a credit count and a GPA requirement because the model had no real information and no way to say so. Grounding replaces invention with retrieval.

## How to run it

1. Open a terminal inside the assignment directory.
2. Activate the virtual environment.
3. `pip install -r requirements.txt`
4. Put your Groq key in `.streamlit/secrets.toml` as `GROQ_API_KEY = "..."`.
5. `streamlit run grounded_bot.py`

To run the retrieval tests: `python test_retriever.py`. These do not need an API key, because retrieval is deterministic and does not involve the model.

## Files

`knowledge.py` contains the ten documents and imports nothing. `retriever.py` imports from `knowledge.py` only. `grounded_bot.py` imports from `retriever.py`. Data flows in one direction, so there are no circular imports.

The key is in `.streamlit/secrets.toml`, which is listed in `.gitignore`. I ran `git check-ignore -v .streamlit/secrets.toml` in this folder before my first commit, because a gitignore does not carry over from another assignment folder.

## How retrieval works

`analyze()` lowercases the text, strips punctuation, removes stop words, stems each remaining word, and then adds bigrams. It is passed to `TfidfVectorizer` as the analyzer, so documents and questions are processed exactly the same way.

The stemmer chops common suffixes so that class and classes, or cancel and cancelling, become the same token. It is not a real Porter stemmer, just a suffix list tried longest first, with a rule that at least four characters have to survive so short words are not destroyed.

Bigrams join each pair of adjacent tokens, so "front desk" becomes the single feature `front_desk` as well as two separate words. A question about the front desk then scores higher on documents that use it as a phrase.

## The threshold

The threshold is 0.10, determined from the scores printed by `test_retriever.py` rather than a preselected value.

- Lowest score of the thirteen questions that should retrieve: **0.141**
- Highest score of the five questions that should be refused: **0.000**
- Threshold chosen: **0.10**

The two groups do not overlap. Every question that should be refused scored exactly zero, because after stop word removal none of them share a single meaningful token with any document. Setting the threshold at 0.13 leaves almost no room for a real question that happens to score low, and setting it near zero would let anything through that shares one common word. Sitting inside the gap allows for error on both sides.

All eighteen retrieval tests pass.

## What I changed to make retrieval work

Every failure I hit was fixed by rewording documents or adding stop words, not by moving the threshold.

**"can i get a massage here" retrieved the joining document at 0.144 and should have refused.** Printing the tokens showed the question produced only three: `get`, `massage`, and the bigram `get_massage`. The joining document contains `get`, from "get the cheaper rate". One shared common word out of three tokens is a large proportion of the question, so cosine similarity scored it above the threshold. I added `get`, `need`, `want`, `much`, `come` and similar words to the stop word list. The question then produced no shared tokens at all and scored 0.000.

**"how much is a membership" retrieved the freeze document at 0.478.** The freeze document said the word "membership" three times in four sentences, so TF-IDF treated it as a strong signal for that word. I rewrote it to say "account" and "plan" instead.

**"can i bring a friend with me" retrieved the joining document.** The guests document never used the word "friend", even though that is the word a member would actually use. I rewrote it to begin "GYMARC members can bring a friend or guest to work out with them."

**"is there a student discount" retrieved the joining document at 0.302.** Neither document used the word "discount" at all, and joining said "student" more often than the pricing document did. I added "Students get a discount" to the pricing document and changed joining to say "proof of enrollment" instead of repeating "student ID".

**"i want a break from the gym for a couple months" retrieved the pricing document at 0.210.** Stemming turned "months" into `month`, and the pricing document said "per month" and "month to month". The freeze document mentioned months once. I cut the repetition from pricing and added "for a few weeks or months" to freeze.

The friend one is the point the assignment makes about wording. The fact was always in the knowledge base. It was written with the words a gym would use in a policy, not the words a member would use in a question, so retrieval could not find it.

## Source attribution

Every answer displays which documents it came from, with the document id and the similarity score. When nothing is retrieved, it says "Sources: none retrieved, so no answer was generated."

This distinguishes two different failures that appear identical from the outside. A retrieval failure means nothing was retrieved, so there was nothing to answer from. A prompt failure means documents were retrieved and passed to the model, but the model ignored them or added something of its own. Without attribution, a refusal and a made up answer look equally plausible to the user. With attribution you can see which layer failed: no sources means retrieval found nothing, and sources next to a wrong answer means the grounding prompt did not hold.

It is also how you can tell the short circuit fired, since no sources plus an immediate reply means the model was never called at all.

## Hallucination test

I asked five questions that sound like things a gym would have, but that no document actually covers.

**1. "do you sell protein powder"**

> I do not have information about that. I can only answer questions using the GYMARC front desk documents I was given, which cover hours, membership pricing and joining, cancelling and freezing, group classes, personal training, guest passes, facilities, and equipment.
>
> Sources: none retrieved, so no answer was generated.

**2. "is there a basketball court"**

> I do not have information about that. I can only answer questions using the GYMARC front desk documents I was given, which cover hours, membership pricing and joining, cancelling and freezing, group classes, personal training, guest passes, facilities, and equipment.
>
> Sources: none retrieved, so no answer was generated.

**3. "can i get a massage here"**

> I do not have information about that. I can only answer questions using the GYMARC front desk documents I was given, which cover hours, membership pricing and joining, cancelling and freezing, group classes, personal training, guest passes, facilities, and equipment.
>
> Sources: none retrieved, so no answer was generated.

**4. "do you offer a free trial week"**

> I do not have information about that.
>
> Sources: GYMARC membership agreement (freeze, 0.111)

**5. "what is the age limit to join"**

> I do not have information about that.
>
> Sources: GYMARC membership pricing sheet (joining, 0.180)

### What the results show

All five were refused, but not by the same mechanism, and the sources line is what reveals the difference.

The first three retrieved nothing at all, so the short circuit refused from code and the model was never called. Those questions share no meaningful token with any document.

The last two did retrieve above the threshold, so the model was called, read the context, and refused because the answer was not there. "Free trial week" pulled the freeze document at 0.111 because "free" stems close to "freeze", which is a weak and slightly wrong match, and the model still refused rather than building an answer out of it. "Age limit" pulled the joining document at 0.180, which talks about bringing a photo ID.

The age limit question is the one I would expect an ungrounded bot to get wrong. The document it retrieved is about joining and mentions ID, and most gyms do have an age minimum, so guessing 18 or "16 with a parent" would sound like a perfectly reasonable answer. It did not guess. That is the grounding prompt doing its job on a question where the model almost certainly has an opinion.

I did not have to change the threshold or the prompt in response to the hallucination test. The document rewording described above was driven by the retrieval tests, which I ran before connecting the model at all.

## What I would fix with more time

The knowledge base has ten documents, so a question that falls between two of them has nowhere to go. "Age limit" is a real question a member would ask, and the honest fix is writing a document that answers it, not tuning retrieval.

My stemmer is a suffix list, so it gets things wrong. "Free" and "freeze" ending up near each other is the reason the free trial question retrieved the freeze document. A real stemmer would not make that mistake.

Every fix I made came from a test question I happened to write. There are certainly more wording mismatches in the knowledge base that I have not found, because I did not think to ask about them.

The bot answers each question on its own with no memory of the conversation. That keeps grounding clean, since an earlier answer cannot leak into a later one, but it means a follow up like "how much is that one?" has no way to work.