import os
import json
from collections import defaultdict


RESULT_DIR = "./eval_results"


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data



def mean_score(data):

    scores = []
    accs = []


    for x in data:

        # OPRD结果字段
        if "score" in x:
            scores.append(x["score"])

        # 兼容旧结果
        elif "reward" in x:
            scores.append(x["reward"])


        if "acc" in x:
            accs.append(x["acc"])



    score = (
        sum(scores) / len(scores)
        if scores else 0
    )


    acc = (
        sum(accs) / len(accs)
        if accs else 0
    )


    return score, acc



def eval_math(data, name):

    score, acc = mean_score(data)


    print("=" * 70)
    print(name)
    print("=" * 70)


    print(
        f"num={len(data)} "
        f"score={score:.4f} "
        f"acc={acc:.4f}"
    )



def eval_code(data, name):


    groups = defaultdict(list)


    for x in data:

        source = x.get(
            "data_source",
            "unknown"
        )

        groups[source].append(x)



    print("=" * 70)
    print(name)
    print("=" * 70)



    total = []


    order = [
        "codeforces",
        "codecontests",
        "taco",
        "apps"
    ]


    for k in order:

        if k in groups:

            score, acc = mean_score(
                groups[k]
            )


            print(
                f"{k:<15}"
                f"num={len(groups[k]):4d} "
                f"score={score:.4f} "
                f"acc={acc:.4f}"
            )


            total.extend(
                groups[k]
            )



    score, acc = mean_score(total)


    print("-" * 70)


    print(
        f"Overall         "
        f"num={len(total):4d} "
        f"score={score:.4f} "
        f"acc={acc:.4f}"
    )



def main():


    files = sorted(
        [
            f
            for f in os.listdir(RESULT_DIR)
            if f.endswith(".jsonl")
        ]
    )


    if len(files) == 0:
        print(
            "No jsonl files found in",
            RESULT_DIR
        )
        return



    for file in files:


        path = os.path.join(
            RESULT_DIR,
            file
        )


        data = load_jsonl(path)



        if "Eurus-Code" in file:

            eval_code(
                data,
                file
            )


        elif (
            "AIME24" in file
            or
            "MATH-500" in file
        ):

            eval_math(
                data,
                file
            )


        else:

            print(
                "Skip:",
                file
            )



if __name__ == "__main__":
    main()