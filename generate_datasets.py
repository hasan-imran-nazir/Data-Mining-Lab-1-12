"""
Generate the CSV datasets used by the lab scripts (standard library only).

Run this once after cloning the repository:
    python generate_datasets.py

The script writes the following files into the "data" folder:
    credit_card_transactions.csv   credit card transactions (labs 2, 3, 6, 7, 8, 11)
    market_basket_transactions.csv market baskets (labs 4, 5)
    customer_segments.csv         customer records with 4 segments (labs 9, 10)
    text_corpus.csv               small text corpus (lab 7)
    student_marks.csv             simple table (lab 1)
"""

import csv
import os
import random

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def write_csv(filename, header, rows):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {filename:35} {len(rows):6d} rows")


def gen_transactions():
    n = 15000
    fraud_idx = set(random.sample(range(n), int(n * 0.006)))
    header = (["Time"] + [f"V{i}" for i in range(1, 29)]
              + ["Amount", "Merchant_Category", "Card_Type", "Country", "Class"])
    merchants = ["Grocery", "Fuel", "Restaurant", "Electronics", "Online", "ATM", "Travel"]
    cards = ["Visa", "MasterCard", "Amex", "Discover"]
    countries = ["PK", "US", "UK", "AE", "SA", "DE", "FR"]
    m_weights = [0.26, 0.18, 0.22, 0.12, 0.12, 0.06, 0.04]
    c_weights = [0.45, 0.33, 0.14, 0.08]
    co_weights = [0.22, 0.26, 0.14, 0.11, 0.09, 0.11, 0.07]

    rows = []
    for i in range(n):
        cls = 1 if i in fraud_idx else 0
        rec = [i * 60]
        for v in range(1, 29):
            mu = 0.0
            if v == 14:
                mu = 1.30 if cls else 0.0
            elif v == 17:
                mu = -1.10 if cls else 0.0
            elif v == 12:
                mu = 0.70 if cls else 0.0
            elif v == 7:
                mu = -0.45 if cls else 0.0
            rec.append(round(random.gauss(mu, 1.0), 6))
        amount = random.lognormvariate(3.0, 1.1)
        if random.random() < 0.004:
            amount *= random.uniform(8.0, 22.0)
        if cls == 1:
            amount *= 0.65
        rec.append(round(amount, 2))
        rec.append(random.choices(merchants, weights=m_weights)[0])
        rec.append(random.choices(cards, weights=c_weights)[0])
        rec.append(random.choices(countries, weights=co_weights)[0])
        rec.append(cls)
        rows.append(rec)

    for rec in rows:
        if random.random() < 0.008:
            rec[2] = ""
        if random.random() < 0.006:
            rec[6] = ""
        if random.random() < 0.004:
            rec[29] = ""

    for _ in range(25):
        rows.append(list(random.choice(rows)))

    random.shuffle(rows)
    write_csv("credit_card_transactions.csv", header, rows)


CATALOG = [
    "Bread", "Butter", "Milk", "Eggs", "Cheese", "Yogurt", "Coffee", "Tea",
    "Sugar", "Flour", "Pasta", "Rice", "Chicken", "Beef", "Tomatoes", "Onions",
    "Potatoes", "Chips", "Cola", "Beer", "Chocolate", "Biscuits", "Diapers",
    "Wipes", "BabyFood", "Soap", "Shampoo", "Toothpaste", "Tissues", "SoapDish",
]

RULES = {
    "Bread": {"Butter": 0.75, "Milk": 0.55, "Eggs": 0.40},
    "Coffee": {"Sugar": 0.65, "Milk": 0.45, "Biscuits": 0.40},
    "Tea": {"Sugar": 0.55, "Biscuits": 0.35},
    "Pasta": {"Tomatoes": 0.65, "Cheese": 0.45, "Onions": 0.35},
    "Chicken": {"Rice": 0.55, "Onions": 0.45, "Potatoes": 0.40},
    "Rice": {"Chicken": 0.50},
    "Chips": {"Cola": 0.80, "Beer": 0.50},
    "Diapers": {"Wipes": 0.75, "BabyFood": 0.45},
    "Butter": {"Bread": 0.70},
    "Milk": {"Eggs": 0.45},
    "Eggs": {"Cheese": 0.40},
    "Chocolate": {"Biscuits": 0.50},
    "Tissues": {"Soap": 0.40},
}


def gen_basket():
    rows = []
    for t in range(1, 251):
        size = random.randint(3, 8)
        items = set()
        seed = random.choice(CATALOG)
        items.add(seed)
        for _ in range(random.randint(1, 4)):
            if random.random() < 0.65 and seed in RULES:
                items.add(random.choice(list(RULES[seed].keys())))
        while len(items) < size:
            items.add(random.choice(CATALOG))
        items = sorted(items)
        rows.append([t, ", ".join(items), len(items)])
    write_csv("market_basket_transactions.csv",
              ["Transaction_ID", "Items", "Num_Items"], rows)


def gen_segments():
    header = ["Customer_ID", "Age", "Annual_Income", "Spending_Score",
              "Purchase_Frequency", "Online_Orders", "Avg_Basket_Size",
              "Recency_Days", "Segment"]
    profiles = [
        ("Young Online Shoppers", 26, 34000, 82, 14.0, 9.0, 3.6, 6),
        ("Mid Loyal Spenders", 45, 68000, 61, 8.0, 3.0, 5.2, 18),
        ("Senior Offline Shoppers", 66, 41000, 27, 2.5, 0.4, 6.1, 55),
        ("High Income Big Spenders", 41, 118000, 88, 11.0, 5.0, 8.4, 10),
    ]
    rows = []
    cid = 1000
    for name, age, income, score, freq, online, basket, rec in profiles:
        for _ in range(60):
            cid += 1
            rows.append([
                cid,
                max(18, min(80, int(random.gauss(age, 6)))),
                max(15000, int(random.gauss(income, income * 0.12))),
                max(1, min(100, int(random.gauss(score, 9)))),
                round(max(0.2, random.gauss(freq, freq * 0.25)), 1),
                round(max(0.0, random.gauss(online, online * 0.3)), 1),
                round(max(1.0, random.gauss(basket, 1.1)), 1),
                max(0, int(random.gauss(rec, rec * 0.4))),
                name,
            ])
    random.shuffle(rows)
    write_csv("customer_segments.csv", header, rows)


CORPUS = [
    ("The team won the championship after a dramatic extra time penalty shootout", "sports"),
    ("He scored two goals in the second half and secured the victory for his club", "sports"),
    ("The match ended in a draw after ninety minutes of intense football action", "sports"),
    ("She broke the world record in the hundred metres final at the championships", "sports"),
    ("The striker scored a hat trick as the team won the league title", "sports"),
    ("Our coach announced the squad for the upcoming tournament matches", "sports"),
    ("The basketball game went into overtime and the crowd went wild", "sports"),
    ("He won the medal at the olympics after the final race of the day", "sports"),
    ("The tennis player defeated the world number one in the grand slam final", "sports"),
    ("The team celebrated the victory with the supporters after the final whistle", "sports"),
    ("She won the championship title in the swimming competition this season", "sports"),
    ("The football club signed a new striker for the coming football season", "sports"),
    ("The senate voted today to approve the new education bill after debate", "politics"),
    ("The president addressed the parliament about the economic recovery plan", "politics"),
    ("Voters went to the polling stations early to cast their ballots today", "politics"),
    ("The election commission announced the final results of the general election", "politics"),
    ("The minister of finance presented the budget to the national assembly", "politics"),
    ("Opposition leaders criticised the new tax policy proposed by the government", "politics"),
    ("The parliament debate continued late into the night over the new law", "politics"),
    ("The election campaign began with rallies in several major cities of the country", "politics"),
    ("The government announced new policies to support the farmers and the economy", "politics"),
    ("The chief election commissioner explained the new voting rules for citizens", "politics"),
    ("The party won the majority of seats in the state legislative assembly", "politics"),
    ("The supreme court heard the case concerning the new constitutional amendment", "politics"),
]


def gen_text():
    write_csv("text_corpus.csv", ["Text", "Label"], CORPUS)


def gen_marks():
    names = ["Ayesha", "Bilal", "Hina", "Usman", "Fatima", "Zain", "Ali", "Sana",
             "Omar", "Maryam", "Hamza", "Areeba"]
    rows = []
    for i, name in enumerate(names):
        ds = random.randint(40, 95)
        de = random.randint(40, 95)
        dm = random.randint(40, 95)
        average = (ds + de + dm) / 3
        rows.append([f"ST{1001 + i}", name, ds, de, dm, round(average, 2),
                     "Pass" if average >= 50 else "Fail"])
    write_csv("student_marks.csv",
              ["Roll_No", "Name", "Data_Science", "DBMS", "Data_Mining", "Average", "Result"],
              rows)


if __name__ == "__main__":
    print("Generating the datasets in", DATA_DIR)
    gen_transactions()
    gen_basket()
    gen_segments()
    gen_text()
    gen_marks()
    print("Done.")
