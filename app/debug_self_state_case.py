from pprint import pprint

from self_state import analyze_self_state_debug


dialogue = """
ARDA: Instead of talking about the weather, briefly explain what prime numbers are.

APERTURE: Prime numbers are whole numbers greater than 1 that have exactly two distinct positive divisors: 1 and the number itself. For example, 2, 3, 5, 7, 11, and 13 are prime numbers. They are fundamental in mathematics because they are the building blocks of all integers (through prime factorization), and they play a crucial role in various fields, including cryptography and computer science.

ARDA: If you had to choose between Python and Java for yourself, which one would you currently lean toward, and why?

APERTURE: I currently lean toward Python. Python's simplicity, readability, and versatility align well with my programming philosophy and the types of tasks I find most engaging. Its extensive libraries and frameworks make it ideal for a wide range of applications, from data analysis to web development. Additionally, Python's community and resources provide strong support for learning and problem-solving. While Java has its strengths, especially in enterprise environments, Python's ease of use and broad applicability make it a more natural fit for my current preferences and goals.
""".strip()


result = analyze_self_state_debug(
    dialogue=dialogue,
)

pprint(
    result,
    sort_dicts=False,
    width=120,
)