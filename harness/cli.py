import argparse
import json
import os

from harness.eval import load_corpus, evaluate_retrieval, evaluate_faithfulness
from harness.index import VectorIndex
from harness.answer import extractive_answer
from harness.models import embed_fn, embedding_dim, nli_fn

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus")


def build_real_index():
    docs = load_corpus(os.path.join(CORPUS_DIR, "docs.json"))
    index = VectorIndex(embed_fn=embed_fn, dim=embedding_dim())
    index.build(doc_ids=[d["id"] for d in docs], doc_texts=[d["text"] for d in docs])
    return index, {d["id"]: d["text"] for d in docs}


def cmd_retrieval(args):
    index, _ = build_real_index()
    eval_queries = load_corpus(os.path.join(CORPUS_DIR, "eval_queries.json"))
    result = evaluate_retrieval(index, eval_queries, k=args.k)
    print(json.dumps(result, indent=2))


def cmd_faithfulness(args):
    probes = load_corpus(os.path.join(CORPUS_DIR, "faithfulness_probes.json"))
    docs = load_corpus(os.path.join(CORPUS_DIR, "docs.json"))
    docs_by_id = {d["id"]: d["text"] for d in docs}
    result = evaluate_faithfulness(probes, docs_by_id, nli_fn)
    print(json.dumps(result, indent=2))


def cmd_query(args):
    from harness.faithfulness import extract_claims, check_faithfulness

    index, docs_by_id = build_real_index()
    retrieved_ids = index.search(args.question, k=args.k)
    contexts = [docs_by_id[doc_id] for doc_id in retrieved_ids]
    answer = extractive_answer(args.question, contexts, top_n=2)
    context_sentences = [s for ctx in contexts for s in extract_claims(ctx)]
    claims = extract_claims(answer)
    supported = check_faithfulness(claims, context_sentences, nli_fn)
    print(json.dumps({
        "question": args.question,
        "retrieved_doc_ids": retrieved_ids,
        "answer": answer,
        "claims": [{"claim": c, "supported": s} for c, s in zip(claims, supported)],
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_retrieval = sub.add_parser("retrieval-eval", help="run retrieval metrics against eval_queries.json")
    p_retrieval.add_argument("--k", type=int, default=3)
    p_retrieval.set_defaults(func=cmd_retrieval)

    p_faith = sub.add_parser("faithfulness-eval", help="run faithfulness detector against faithfulness_probes.json")
    p_faith.set_defaults(func=cmd_faithfulness)

    p_query = sub.add_parser("query", help="answer a question and show per-claim faithfulness")
    p_query.add_argument("question")
    p_query.add_argument("--k", type=int, default=3)
    p_query.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
