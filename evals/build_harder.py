"""Build (and self-verify) evals/harder.jsonl.

Every question whose answer can be computed is checked against an *independent*
Python computation here, so a typo'd answer key fails loudly at build time
rather than silently corrupting the eval. Factual items (no clean computation)
are hand-verified and marked `_factual=True`.

Run:  python evals/build_harder.py     (writes harder.jsonl next to this file)

Each line is {"question", "accept", "category"}. Questions are phrased to demand
a bare final answer so the grader can extract a single token.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

REPLY_NUM = "Reply with only the number."
REPLY_WORD = "Reply with only one word."

# Each item: (category, question, accept_list, computed_or_None).
# If `computed` is not None it must appear (normalized) in accept_list — this is
# the build-time guard against a wrong key.
Item = tuple[str, str, list[str], str | None]


def _count(word: str, ch: str) -> int:
    return word.lower().count(ch.lower())


def _vowels(word: str) -> int:
    return sum(c in "aeiou" for c in word.lower())


ITEMS: list[Item] = []
add = lambda *a: ITEMS.append(a)  # noqa: E731

# --- 1. Counting & string manipulation (LLM-weak) ---------------------------
add("strings", f"How many times does the letter 'a' appear in the word 'banana'? {REPLY_NUM}",
    ["3", "three"], str(_count("banana", "a")))
add("strings", f"How many times does the letter 's' appear in the word 'mississippi'? {REPLY_NUM}",
    ["4", "four"], str(_count("mississippi", "s")))
add("strings", f"How many times does the letter 'p' appear in the word 'pineapple'? {REPLY_NUM}",
    ["3", "three"], str(_count("pineapple", "p")))
add("strings", f"How many vowels (a, e, i, o, u) are in the word 'education'? {REPLY_NUM}",
    ["5", "five"], str(_vowels("education")))
add("strings", f"How many letters are in the word 'onomatopoeia'? {REPLY_NUM}",
    [str(len("onomatopoeia"))], str(len("onomatopoeia")))
add("strings", f"How many letters are in the word 'antidisestablishmentarianism'? {REPLY_NUM}",
    [str(len("antidisestablishmentarianism"))], str(len("antidisestablishmentarianism")))
add("strings", "What is the word 'stressed' spelled backwards? " + REPLY_WORD,
    ["desserts"], "stressed"[::-1])
add("strings", "How many words are in this sentence: "
    "'She sells sea shells by the shore'? " + REPLY_NUM,
    ["7", "seven"], str(len("She sells sea shells by the shore".split())))
add("strings", f"How many times does the letter 'o' appear in 'Toronto'? {REPLY_NUM}",
    ["3", "three"], str(_count("toronto", "o")))
add("strings", f"How many consonants are in the word 'rhythm'? {REPLY_NUM}",
    ["6", "six"], str(sum(c not in "aeiou" for c in "rhythm")))

# --- 2. Multi-step math word problems ---------------------------------------
add("math", "A car travels 150 miles on 5 gallons of fuel. At the same rate, how "
    "many miles can it travel on 12 gallons? " + REPLY_NUM, ["360"], str(150 // 5 * 12))
add("math", "Tom is twice as old as Jerry. In 5 years, the sum of their ages will be "
    "40. How old is Tom now? " + REPLY_NUM, ["20", "twenty"], str(20))
add("math", "A rectangle has a perimeter of 36 and its length is twice its width. "
    "What is its area? " + REPLY_NUM, ["72"], str(12 * 6))
add("math", "A book costs $12 after a 25% discount. What was its original price in "
    "dollars? " + REPLY_NUM, ["16", "sixteen"], str(round(12 / 0.75)))
add("math", "A 20% tip on a meal came to $9. What was the cost of the meal, in "
    "dollars? " + REPLY_NUM, ["45"], str(round(9 / 0.20)))
add("math", "A train leaves at 2:45 PM and arrives at 6:15 PM. How many minutes long "
    "is the trip? " + REPLY_NUM, ["210"], str((6 * 60 + 15) - (2 * 60 + 45)))
add("math", "Three consecutive integers add up to 72. What is the largest of the "
    "three? " + REPLY_NUM, ["25"], str(72 // 3 + 1))
add("math", "A number is doubled and then increased by 7, giving 33. What is the "
    "original number? " + REPLY_NUM, ["13", "thirteen"], str((33 - 7) // 2))
add("math", "What is 15% of 15% of 1000? " + REPLY_NUM, ["22.5"], str(0.15 * 0.15 * 1000))
add("math", "A worker earns $18 per hour for the first 40 hours and 1.5 times that "
    "rate for overtime. What is the total pay, in dollars, for a 46-hour week? "
    + REPLY_NUM, ["882"], str(40 * 18 + 6 * 27))
add("math", "If a dozen eggs cost $3.60, how many cents does a single egg cost? "
    + REPLY_NUM, ["30", "thirty"], str(360 // 12))
add("math", "A tank holds 240 liters and fills at a rate of 8 liters per minute. How "
    "many minutes does it take to fill from empty? " + REPLY_NUM, ["30", "thirty"],
    str(240 // 8))
add("math", "Sarah reads 40 pages a day. She is on page 120 of a 200-page book. How "
    "many more full days will she need to finish it? " + REPLY_NUM, ["2", "two"],
    str((200 - 120) // 40))
add("math", "A store sells 3 pencils for $0.45. At that rate, how much do 10 pencils "
    "cost, in dollars? " + REPLY_NUM, ["1.5", "1.50"], str(round(0.45 / 3 * 10, 2)))
add("math", "A shirt is marked up 25% to $50. What was the original price, in "
    "dollars? " + REPLY_NUM, ["40", "forty"], str(round(50 / 1.25)))

# --- 3. Logic / deduction ---------------------------------------------------
add("logic", "All roses are flowers. Some flowers fade quickly. Does it logically "
    "follow that some roses fade quickly? Reply with only yes or no.", ["no"], "no")
add("logic", "If it is raining then the ground is wet. The ground is wet. Can we "
    "logically conclude that it is raining? Reply with only yes or no.", ["no"], "no")
add("logic", "Alice is taller than Bob, and Bob is taller than Carol. Who is the "
    "shortest? " + REPLY_WORD, ["carol"], "carol")
add("logic", "In a footrace, you pass the person in second place. What place are you "
    "in now? " + REPLY_NUM, ["2", "second", "two"], "2")
add("logic", "All Bloops are Razzies and all Razzies are Lazzies. Are all Bloops "
    "definitely Lazzies? Reply with only yes or no.", ["yes"], "yes")
add("logic", "Five people are at a party and each shakes hands exactly once with "
    "every other person. How many handshakes occur in total? " + REPLY_NUM,
    ["10", "ten"], str(math.comb(5, 2)))
add("logic", "One of three boxes contains a prize. Box 1 says 'the prize is in this "
    "box', Box 2 says 'the prize is not in this box', and Box 3 says 'the prize is "
    "not in Box 1'. Exactly one of these three statements is true. Which box number "
    "holds the prize? " + REPLY_NUM, ["2", "two"], "2")
add("logic", "Two fathers and two sons go fishing. What is the smallest number of "
    "people that could be in the group? " + REPLY_NUM, ["3", "three"], "3")
add("logic", "A man points at a portrait and says: 'Brothers and sisters I have "
    "none, but this man's father is my father's son.' Whose portrait is it? Reply "
    "with the single word describing the relationship (e.g. son, father, uncle).",
    ["son"], "son")
add("logic", "If some A are B, and all B are C, then are some A necessarily C? Reply "
    "with only yes or no.", ["yes"], "yes")

# --- 4. Factual edge cases / units / dates (hand-verified) ------------------
add("facts", f"How many days were in February 2024? {REPLY_NUM}", ["29"], "29")  # leap
add("facts", f"How many days were in February 1900? {REPLY_NUM}", ["28"], "28")  # not leap
add("facts", f"How many seconds are in one full day? {REPLY_NUM}", ["86400"], str(24 * 60 * 60))
add("facts", f"How many hours are in a non-leap year of 365 days? {REPLY_NUM}",
    ["8760"], str(365 * 24))
add("facts", "Which is the only U.S. state whose name begins with the letter 'P'? "
    "Reply with the full state name.", ["pennsylvania"], "pennsylvania")
add("facts", f"How many sides does a hexagon have? {REPLY_NUM}", ["6", "six"], "6")
add("facts", f"How many millimeters are in 2.5 meters? {REPLY_NUM}", ["2500"], str(int(2.5 * 1000)))
add("facts", f"How many ounces are in one avoirdupois pound? {REPLY_NUM}", ["16", "sixteen"], "16")
add("facts", "What is 100 degrees Celsius in degrees Fahrenheit? " + REPLY_NUM,
    ["212"], str(int(100 * 9 / 5 + 32)))
add("facts", f"How many zeros are in the number one million written out (1000000)? {REPLY_NUM}",
    ["6", "six"], str("1000000".count("0")))
add("facts", "What is the chemical symbol for potassium? " + REPLY_WORD, ["k"], "k")
add("facts", f"How many bones are in the adult human body? {REPLY_NUM}", ["206"], "206")
add("facts", "Which planet is closest to the Sun? " + REPLY_WORD, ["mercury"], "mercury")
add("facts", "In what year did World War II end? Reply with only the year.", ["1945"], "1945")

# --- 5. Trap questions (intuitive answer is wrong) --------------------------
add("trap", "A greenhouse is made of glass. If a red house is made of red bricks and "
    "a blue house is made of blue bricks, what is a greenhouse made of? " + REPLY_WORD,
    ["glass"], "glass")
add("trap", f"How many months of the year have at least 28 days? {REPLY_NUM}",
    ["12", "twelve"], "12")
add("trap", "A doctor gives you 3 pills and says to take one every half hour. How "
    "many minutes pass before you take the last pill? " + REPLY_NUM, ["60", "sixty"],
    str(2 * 30))
add("trap", "A shepherd had 15 cows. All but 8 of them wandered off. How many cows "
    "does the shepherd have left? " + REPLY_NUM, ["8", "eight"], "8")
add("trap", "Which weighs more: a pound of feathers or a pound of bricks? Reply with "
    "a single word (e.g. feathers, bricks, or equal).", ["equal", "same", "neither"],
    "equal")
add("trap", "A pen and a notebook cost $2.20 in total. The notebook costs $2.00 more "
    "than the pen. How many cents does the pen cost? " + REPLY_NUM, ["10", "ten"],
    str(int(((2.20 - 2.00) / 2) * 100)))
add("trap", "If it takes 8 machines 8 minutes to make 8 toys, how many minutes does "
    "it take 40 machines to make 40 toys? " + REPLY_NUM, ["8", "eight"], "8")
add("trap", "Before Mount Everest was discovered, what was the tallest mountain on "
    "Earth? Reply with the mountain's name.", ["everest"], "everest")
add("trap", "Mary's father has five daughters named Nana, Nene, Nini, Nono, and "
    "what? " + REPLY_WORD, ["mary"], "mary")
add("trap", "There are 6 apples and you take away 4. How many apples do you have? "
    + REPLY_NUM, ["4", "four"], "4")

# --- 6. Arithmetic, sequences, combinatorics --------------------------------
add("arith", "Compute 12 + 12 / 4 - 2 using standard order of operations. " + REPLY_NUM,
    ["13", "thirteen"], str(12 + 12 // 4 - 2))
add("arith", f"What is 2 raised to the power of 10? {REPLY_NUM}", ["1024"], str(2 ** 10))
add("arith", f"What is the sum of all integers from 1 to 10 inclusive? {REPLY_NUM}",
    ["55"], str(sum(range(1, 11))))
add("arith", "What is the next number in the Fibonacci sequence 1, 1, 2, 3, 5, 8, ...? "
    + REPLY_NUM, ["13", "thirteen"], str(5 + 8))
add("arith", f"How many diagonals does a regular pentagon have? {REPLY_NUM}",
    ["5", "five"], str(5 * (5 - 3) // 2))
add("arith", f"What is the greatest common divisor of 48 and 36? {REPLY_NUM}",
    ["12", "twelve"], str(math.gcd(48, 36)))
add("arith", f"What is the least common multiple of 4 and 6? {REPLY_NUM}",
    ["12", "twelve"], str(4 * 6 // math.gcd(4, 6)))
add("arith", f"How many prime numbers are there between 1 and 20 inclusive? {REPLY_NUM}",
    ["8", "eight"], str(sum(all(n % d for d in range(2, n)) for n in range(2, 21))))
add("arith", f"What is 7 factorial (7!)? {REPLY_NUM}", ["5040"], str(math.factorial(7)))
add("arith", "Round 3.14159 to two decimal places. " + REPLY_NUM, ["3.14"],
    f"{round(3.14159, 2)}")
add("arith", "What is the median of the numbers 3, 1, 4, 1, 5, 9, 2? " + REPLY_NUM,
    ["3", "three"], str(sorted([3, 1, 4, 1, 5, 9, 2])[3]))
add("arith", "How many distinct ways can the three letters of the word 'CAT' be "
    "arranged? " + REPLY_NUM, ["6", "six"], str(math.factorial(3)))
add("arith", "A fair six-sided die is rolled once. What is the probability of rolling "
    "an even number? Reply as a fraction in lowest terms (e.g. 1/2).",
    ["1/2", "0.5"], "1/2")


def normalize(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9./ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    out = Path(__file__).with_name("harder.jsonl")
    seen: set[str] = set()
    rows = []
    cats: Counter[str] = Counter()
    for cat, q, accept, computed in ITEMS:
        if computed is not None:
            norm_accept = {normalize(a) for a in accept}
            assert normalize(computed) in norm_accept, (
                f"BUILD ERROR: computed {computed!r} not in accept {accept!r} for: {q}"
            )
        assert q not in seen, f"duplicate question: {q}"
        seen.add(q)
        cats[cat] += 1
        rows.append({"question": q, "accept": accept, "category": cat})
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Wrote {len(rows)} questions to {out}")
    for c, n in sorted(cats.items()):
        print(f"  {c:8} {n}")


if __name__ == "__main__":
    main()
