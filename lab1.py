"""
Lab 1: Introduction to Python for Data Mining
Topics:
  Python basics (comments, print, indentation)
  Variables, data types and operators
  Lists, tuples and dictionaries
  Functions and loops
  Reading and writing CSV files using Pandas
Run:  python <this file>.py
"""

from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print("LAB 1 : INTRODUCTION TO PYTHON FOR DATA MINING")
print("=" * 78)


def section(title: str) -> None:
    """Print a nice section header."""
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


# ==================================================================================
# 1. PYTHON BASICS
# ==================================================================================
section("1. PYTHON BASICS")

print("Python is interpreted, dynamically typed and indentation based.")
print("A '#' starts a comment and every block is defined by its indentation.")

print("\nHello, Data Mining!")


# ==================================================================================
# 2. VARIABLES, DATA TYPES AND OPERATORS
# ==================================================================================
section("2. VARIABLES, DATA TYPES AND OPERATORS")

name = "Imran"
roll_no = 21
cgpa = 3.72
is_student = True
subjects = ["Data Mining", "DBMS"]
result = None

print(f"name       = {name!r:12} type = {type(name)}")
print(f"roll_no    = {roll_no!r:12} type = {type(roll_no)}")
print(f"cgpa       = {cgpa!r:12} type = {type(cgpa)}")
print(f"is_student = {is_student!r:12} type = {type(is_student)}")
print(f"subjects   = {subjects!r:12} type = {type(subjects)}")
print(f"result     = {result!r:12} type = {type(result)}")

cgpa_int = int(cgpa)
cgpa_str = str(cgpa)
cgpa_float = float("3.72")
print(f"\nType casting : {cgpa} -> {cgpa_int} ({type(cgpa_int)})")
print(f"Type casting : {cgpa} -> {cgpa_str!r} ({type(cgpa_str)})")
print(f"Type casting : '3.72' -> {cgpa_float} ({type(cgpa_float)})")

a, b = 17, 5
print("\nArithmetic operators")
print(f"a + b  = {a + b}     (addition)")
print(f"a - b  = {a - b}     (subtraction)")
print(f"a * b  = {a * b}     (multiplication)")
print(f"a / b  = {a / b:.4f}  (true division)")
print(f"a // b = {a // b}     (floor division)")
print(f"a % b  = {a % b}      (modulus / remainder)")
print(f"a ** b = {a ** b}     (exponentiation)")

print("\nComparison operators")
print(f"a == b -> {a == b}   a != b -> {a != b}")
print(f"a > b  -> {a > b}    a < b  -> {a < b}")
print(f"a >= b -> {a >= b}   a <= b -> {a <= b}")

print("\nLogical operators")
print(f"True and False      -> {True and False}")
print(f"True or False       -> {True or False}")
print(f"not True            -> {not True}")

print("\nMembership and identity operators")
print(f"'Data Mining' in subjects          -> {'Data Mining' in subjects}")
print(f"'AI' in subjects                   -> {'AI' in subjects}")
print(f"subjects is subjects               -> {subjects is subjects}")
print(f"subjects == ['Data Mining', 'DBMS'] -> {subjects == ['Data Mining', 'DBMS']}")

x = 10
x += 5
x *= 2
print(f"\nx after 'x += 5' and 'x *= 2' -> {x}")
print(f"5 & 3 = {5 & 3}, 5 | 3 = {5 | 3}, 5 ^ 3 = {5 ^ 3}, ~5 = {~5}")
print(f"Ternary operator -> {'Pass' if cgpa >= 3 else 'Fail'}")


# ==================================================================================
# 3. LISTS, TUPLES AND DICTIONARIES
# ==================================================================================
section("3. LISTS, TUPLES AND DICTIONARIES")

marks = [88, 92, 75, 68, 95]
print(f"Original list          : {marks}")
print(f"Indexing  marks[0]      : {marks[0]}")
print(f"Slicing   marks[1:4]    : {marks[1:4]}")
marks.append(90)
marks.insert(0, 100)
print(f"After append/insert    : {marks}")
marks.remove(100)
marks.sort()
print(f"After sort             : {marks}")
marks.reverse()
print(f"After reverse          : {marks}")
print(f"len(marks) = {len(marks)}, sum = {sum(marks)}, max = {max(marks)}, min = {min(marks)}")
print(f"Mean of marks          : {sum(marks) / len(marks):.2f}")

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(f"Nested list            : {matrix}")
print(f"matrix[1][2]           : {matrix[1][2]}")

coordinates = (24.3566, 67.0011)
point = (10, 20, 30)
print(f"\nTuple                  : {coordinates}")
print(f"Type of tuple          : {type(coordinates)}")
print(f"Unpacking              : lat, lon = coordinates -> {coordinates}")
try:
    coordinates[0] = 99
except TypeError as err:
    print(f"Tuples are immutable   : {err}")

student = {
    "Roll_No": "ST1001",
    "Name": "Ayesha Khan",
    "CGPA": 3.72,
    "Courses": ["Data Mining", "DBMS", "AI"],
    "Passed": True,
}
print(f"\nDictionary keys        : {list(student.keys())}")
print(f"Dictionary values      : {list(student.values())}")
print(f"student['CGPA']        : {student['CGPA']}")
student["University"] = "NED University"
student["CGPA"] = 3.85
print(f"After update           : {student}")
print(f"Using get() (safe)     : {student.get('Hostel', 'Not provided')}")
removed = student.pop("University")
print(f"pop('University')      : {removed} | remaining keys: {list(student.keys())}")

marks_dict = {f"Student{i + 1}": m for i, m in enumerate([88, 92, 75])}
print(f"Dict comprehension     : {marks_dict}")

unique_courses = {"Data Mining", "DBMS", "AI", "Data Mining"}
print(f"\nSet (duplicates removed): {unique_courses}")
print(f"Set operations        : A|B = {unique_courses | {'Python'}}, A&B = {unique_courses & {'AI', 'ML'}}")

grades = [("A", 88), ("B", 92), ("A", 75), ("C", 95)]
grouped: dict[str, list] = {}
for grade, mark in grades:
    grouped.setdefault(grade, []).append(mark)
print(f"Grouping with a dict   : {grouped}")


# ==================================================================================
# 4. FUNCTIONS AND LOOPS
# ==================================================================================
section("4. FUNCTIONS AND LOOPS")


def greet(student_name: str, greeting: str = "Hello") -> str:
    """Simple function with a default argument."""
    return f"{greeting}, {student_name}!"


def grade_of(mark: float, passing: float = 50.0) -> str:
    """Return Pass / Fail based on the mark."""
    return "Pass" if mark >= passing else "Fail"


def summarize(*numbers, **extra):
    """Function with variable positional and keyword arguments."""
    total = sum(numbers)
    average = total / len(numbers) if numbers else 0
    return {"total": total, "average": round(average, 2), "extra": extra}


def fibonacci(n: int) -> list:
    """Return the first n Fibonacci numbers (iterative, O(n))."""
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


print(greet("Imran"))
print(greet("Ayesha", greeting="Assalam-o-Alaikum"))
print(f"grade_of(88)  -> {grade_of(88)}")
print(f"grade_of(35)  -> {grade_of(35)}")
print(f"summarize(88, 92, 75) -> {summarize(88, 92, 75, subject='Data Mining')}")
print(f"fibonacci(10) -> {fibonacci(10)}")

square = lambda n: n ** 2
print(f"lambda square(7) -> {square(7)}")
print(f"lambda sorted with key -> {sorted([(2, 'b'), (1, 'a')], key=lambda t: t[0])}")

section("4.1 LOOPS")
print("For loop over a range:")
total = 0
for i in range(1, 11):
    total += i
print(f"Sum of 1..10 = {total}")

print("\nFor loop over a list:")
for idx, subject in enumerate(subjects, start=1):
    print(f"  {idx}. {subject}")

print("\nNested loop (multiplication table of 3):")
for i in range(1, 4):
    print("  " + " ".join(f"{i * j:3}" for j in range(1, 4)))

print("\nZip two lists together:")
for name_, mark in zip(["Ayesha", "Bilal", "Hina"], [88, 92, 75]):
    print(f"  {name_} -> {mark}")

count = 0
print("\nWhile loop:")
while count < 5:
    print(f"  iteration {count}")
    count += 1

print("\nbreak and continue:")
for n in range(1, 11):
    if n == 5:
        print("  break at 5")
        break
    if n % 2 == 0:
        continue
    print(f"  odd value -> {n}")

squares = [n ** 2 for n in range(1, 11) if n % 2 == 0]
print(f"\nList comprehension (even squares) : {squares}")
flat = [n ** 2 for n in range(1, 6)]
print(f"List comprehension                : {flat}")
gen = (n ** 2 for n in range(1, 6))
print(f"Generator expression              : {list(gen)}")


# ==================================================================================
# 5. READING AND WRITING CSV FILES USING PANDAS
# ==================================================================================
section("5. READING AND WRITING CSV FILES USING PANDAS")

csv_file = DATA_DIR / "student_marks.csv"

print(f"Reading file: {csv_file}")
df = pd.read_csv(csv_file)

print("\nFirst 5 rows (head):")
print(df.head())

print("\nLast 2 rows (tail):")
print(df.tail(2))

print("\nShape (rows, columns):", df.shape)
print("\nColumns:", list(df.columns))
print("\nData types (dtypes):")
print(df.dtypes)

print("\ninfo():")
df.info()

print("\ndescribe() - descriptive statistics:")
print(df.describe())

print("\nSingle column (df['Name']):")
print(df["Name"].head())

print("\nMultiple columns (df[['Name', 'Average']]):")
print(df[["Name", "Average"]].head())

print("\nRows selected with a boolean condition (Average >= 80):")
print(df[df["Average"] >= 80][["Roll_No", "Name", "Average"]])

print("\nRows selected with .loc / .iloc:")
print(df.loc[0:2, ["Name", "Data_Mining"]])
print(df.iloc[[0, 1], [1, 4]])

df["Grade"] = df["Average"].apply(lambda m: "A" if m >= 85 else ("B" if m >= 70 else "C"))
print("\nDataFrame after adding a derived column 'Grade':")
print(df.head())

print("\nGroup by Result -> count and average:")
grouped_result = df.groupby("Result").agg(
    Students=("Roll_No", "count"),
    Average_Marks=("Average", "mean"),
)
print(grouped_result)

print("\nSorted by Average descending:")
print(df.sort_values(by="Average", ascending=False)[["Roll_No", "Name", "Average", "Grade"]].head())

out_file = OUT_DIR / "lab1_pass_students.csv"
df[df["Result"] == "Pass"].to_csv(out_file, index=False)
print(f"\nFiltered rows written to : {out_file}")
print(f"Rows written              : {len(df[df['Result'] == 'Pass'])}")

summary_file = OUT_DIR / "lab1_summary.csv"
grouped_result.to_csv(summary_file)
print(f"Summary table written to  : {summary_file}")

check = pd.read_csv(out_file)
print("\nReading the written file back:")
print(check.head())

print("\n" + "=" * 78)
print("END OF LAB 1")
print("=" * 78)
