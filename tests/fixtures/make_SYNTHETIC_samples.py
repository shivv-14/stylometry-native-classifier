"""Generate tests/fixtures/SYNTHETIC_samples.csv.

SYNTHETIC data for unit tests only. The texts are made from templates; they are not
real writing, carry no information about real native or non-native writers, and
must never be used to report results.
"""
import csv
import random
from pathlib import Path

A_SENTENCES = [
    "Part-time work, which many students take on, can teach responsibility; however, it may also cost them sleep.",
    "In my view, the question is not whether students should work, but how much they can manage.",
    "Restaurants that allow smoking, for instance, expose staff to risks they did not choose.",
    "Admittedly, a small income helps; yet the long-term value of study is harder to measure.",
    "It is worth asking whether the benefits outweigh the costs, and, if so, for whom.",
    "Employers, on the other hand, often prefer workers who have already shown some independence.",
]
B_SENTENCES = [
    "I think student should do part time job because it is good for money.",
    "Smoking is bad for health so it must be banned in all restaurant in the country.",
    "Many student in my university do job and they get experience about society.",
    "But if they work too much then they can not study well and the grade will down.",
    "So I agree with this opinion and I want to say the government must make rule.",
    "Also the people who does not smoke can be happy when they eat the food.",
]


def essay(pool, rng, n_sent=12):
    return " ".join(rng.choice(pool) for _ in range(n_sent))


def main():
    rng = random.Random(0)
    rows = []
    for w in range(12):
        for topic in ("PTJ", "SMK"):
            rows.append({"text": essay(A_SENTENCES, rng), "label": "native", "writer_id": f"SYN_A{w:02d}",
                         "topic": topic, "l1": "SYN_A", "proficiency": "XX"})
            rows.append({"text": essay(B_SENTENCES, rng), "label": "non_native", "writer_id": f"SYN_B{w:02d}",
                         "topic": topic, "l1": rng.choice(["SYN_B", "SYN_C"]), "proficiency": rng.choice(["B1", "B2"])})
    out = Path(__file__).with_name("SYNTHETIC_samples.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} SYNTHETIC rows to {out}")


if __name__ == "__main__":
    main()
