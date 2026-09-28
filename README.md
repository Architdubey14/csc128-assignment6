# GYMARC Grounded Bot

CSC-128 Assignment 6

Archit Dubey

## Overview

A Streamlit bot answering questions about GYMARC with only a knowledge base I wrote. The user asks something and a TF-IDF retriever finds the chunks to include, then those chunks are the only things passed to the language model. If nothing is retrieved then the model is never called and the bot refuses.

This is the solution to the problem in Assignment 5, where my bot invented a credit count and a GPA requirement because the model had no real information and no way to say so. Grounding replaces invention with retrieval.

## How to run it

1. Open a terminal inside the assignment directory.
2. Activate the virtual environment.
3. `pip install -r requirements.txt`
4. Put your Groq key in `.streamlit/secrets.toml` as `GROQ_API_KEY = "..."`.
5. `streamlit run grounded_bot.py`

To run the retrieval tests: `python test_retriever.py`. These do not need an API key, because retrieval is deterministic and does not involve the model.

## Files

`knowledge.py` contains the ten chunks and imports nothing. `retriever.py` imports from `knowledge.py` only. `grounded_bot.py` imports from `retriever.py`. Data flows in one direction, so there are no circular imports.

The key is in `.streamlit/secrets.toml`, which is listed in `.gitignore`. I ran `git check-ignore -v .streamlit/secrets.toml` in this folder before my first commit, because a gitignore does not carry over from another assignment folder.

## The threshold

The threshold is 0.10, determined from the scores printed by `test_retriever.py` rather than a preselected value.

- Lowest score of the thirteen questions that should retrieve: **0.201**
- Highest score of the five questions that should be refused: **0.000**
- Threshold chosen: **0.10**

The two groups do not overlap at all. Every question that should be refused scored exactly zero, because none of them shared any meaningful word with any chunk. Setting the threshold at 0.19 leaves little room for a real question that accidentally scores low, and setting it at 0.01 allows anything that shares a word to come through. Sitting in the middle of the gap allows for error on both sides.

All eighteen retrieval tests pass.

## What I changed to make retrieval work

Three questions failed on the first run, and all three were fixed by rewording chunks rather than changing the threshold.

**"how much is a membership" returned the freeze chunk at 0.478.** The freeze chunk said the word "membership" three times in four sentences, so TF-IDF treated it as a strong signal for that word. The pricing chunk uses the word too, but it is longer and more varied, so the term was diluted there. I rewrote the freeze chunk to say "account" and "plan" instead.

**"how much is a membership" then returned the cancel chunk at 0.311.** The same problem in a different chunk. I changed "stop your membership this same way" to "stop coming this same way", removing one mention without losing meaning.

**"can i bring a friend with me" returned the joining chunk.** The guests chunk never used the word "friend", even though that is the word a member would actually use. I rewrote it to start "GYMARC members can bring a friend or guest to work out with them."

That last one is the point the assignment makes about chunk wording. The fact was always in the knowledge base. It was written with the words a gym would use in a policy, not the words a member would use in a question, so retrieval could not find it.

## Source attribution

Every answer displays which documents it came from, with the chunk id and the similarity score. When nothing is retrieved, it says "Sources: none retrieved, so no answer was generated."

This distinguishes two different failures that appear identical from the outside. A retrieval failure means nothing was retrieved, so there was nothing to answer from. A prompt failure means chunks were retrieved and passed to the model, but the model ignored them or added something of its own. Without attribution, a refusal and a made up answer look equally plausible to the user. With attribution you can see which layer failed: no sources means retrieval found nothing, and sources next to a wrong answer means the grounding prompt did not hold.

It is also how you can tell the short circuit fired, since no sources plus an immediate reply means the model was never called at all.

## Hallucination test

I asked five questions that sound like things a gym would have, but that no chunk actually covers.

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
> Sources: GYMARC facility guide (facilities, 0.182); GYMARC facility guide (equipment, 0.154); GYMARC guest policy (guests, 0.131)

**5. "what is the age limit to join"**

> I do not have information about that.
>
> Sources: GYMARC membership pricing sheet (joining, 0.266)

### What the results show

All five were refused, but not by the same mechanism, and the sources line is what reveals the difference.

The first three retrieved nothing at all, so the short circuit refused from code and the model was never called. Those questions share no meaningful word with any chunk.

The last two did retrieve chunks above the threshold, so the model was called, read the context, and refused because the answer was not there. "Free trial week" pulled the facilities and guests chunks because of the words "free" and "pass", but neither chunk says anything about a trial. "Age limit" pulled the joining chunk at 0.266, which talks about bringing a photo ID.

The age limit question is the one I would expect an ungrounded bot to get wrong. The chunk it retrieved is about joining and mentions ID, and most gyms do have an age minimum, so guessing 18 or "16 with a parent" would sound like a perfectly reasonable answer. It did not guess. That is the grounding prompt doing its job on a question where the model almost certainly has an opinion.

I did not have to change the threshold, the chunks, or the prompt in response to the hallucination test. The chunk rewording described above was driven by the retrieval tests, which I ran before connecting the model at all.

## What I would fix with more time

The knowledge base has ten chunks, so a question that falls between two of them has nowhere to go. "Age limit" is a real question a member would ask, and the honest fix is writing a chunk that answers it, not tuning retrieval.

Retrieval matches words, so a question phrased in a completely different vocabulary still fails even when the fact exists. I found three of those and fixed them by rewriting chunks, but I only found them because I happened to write those test questions. There are definitely more.

The bot answers each question on its own with no memory of the conversation. That keeps grounding clean, since an earlier answer cannot leak into a later one, but it means a follow up like "how much is that one?" has no way to work.