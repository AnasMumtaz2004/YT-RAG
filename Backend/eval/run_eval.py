import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel
from langchain_openai import ChatOpenAI

import rag
from transcript import get_transcript

K = 4
judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class Faithfulness(BaseModel):
    supported: bool
    reason: str


class Correctness(BaseModel):
    correct: bool
    reason: str


faith_judge = judge.with_structured_output(Faithfulness)
correct_judge = judge.with_structured_output(Correctness)


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def check_faithfulness(answer, context):
    return faith_judge.invoke(
        f"""Context:
{context}

Answer:
{answer}

Is every factual claim in the answer supported by the context?
If the answer says it doesn't know, that counts as supported."""
    )


def check_correctness(question, answer, reference):
    return correct_judge.invoke(
        f"""Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Does the candidate convey the same facts as the reference? Ignore wording and language."""
    )


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    phrases = ["don't know", "do not know", "not in the", "isn't mentioned",
               "not mentioned", "cannot find", "can't find"]
    return any(p in a for p in phrases)


def main():
    data = json.loads(Path(__file__).with_name("dataset.json").read_text(encoding="utf-8"))

    for vid in {d["video_id"] for d in data}:
        if not rag.has_video(vid):
            print(f"Ingesting {vid}...")
            rag.ingest(vid, get_transcript(vid))

    rows = []
    hits = n_ret = 0
    rr_sum = 0.0
    faith_ok = correct_ok = n_ans = 0
    refuse_ok = n_unans = 0

    for item in data:
        # allow_full=False so retrieval is actually tested, even on short videos
        answer, docs = rag.ask(item["video_id"], item["question"], K, allow_full=False)
        context = "\n\n".join(d.page_content for d in docs)
        row = {"video_id": item["video_id"], "question": item["question"], "answer": answer}

        if item["answerable"]:
            n_ans += 1
            gold = norm(item["gold_snippet"])
            rank = next(
                (i + 1 for i, d in enumerate(docs) if gold in norm(d.page_content)), None
            )
            n_ret += 1
            hits += rank is not None
            rr_sum += 1 / rank if rank else 0
            row["gold_rank"] = rank

            f = check_faithfulness(answer, context)
            c = check_correctness(item["question"], answer, item["reference"])
            faith_ok += f.supported
            correct_ok += c.correct
            row.update(
                faithful=f.supported, correct=c.correct,
                faith_reason=f.reason, correct_reason=c.reason,
            )
        else:
            n_unans += 1
            ok = is_refusal(answer)
            refuse_ok += ok
            row["refused_correctly"] = ok

        rows.append(row)

    def ratio(a, b):
        return round(a / b, 3) if b else None

    summary = {
        "k": K,
        "hit_rate": ratio(hits, n_ret),
        "mrr": ratio(rr_sum, n_ret),
        "faithfulness": ratio(faith_ok, n_ans),
        "correctness": ratio(correct_ok, n_ans),
        "refusal_accuracy": ratio(refuse_ok, n_unans),
    }

    Path(__file__).with_name("results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))

    print("\nFailures:")
    for r in rows:
        if "gold_rank" in r and r["gold_rank"] is None:
            print(f"[RETRIEVAL MISS] {r['question']}")
        elif r.get("correct") is False:
            print(f"[WRONG] {r['question']} -> {r['correct_reason']}")
        elif r.get("faithful") is False:
            print(f"[HALLUCINATION] {r['question']} -> {r['faith_reason']}")
        elif r.get("refused_correctly") is False:
            print(f"[SHOULD HAVE REFUSED] {r['question']}")


if __name__ == "__main__":
    main()