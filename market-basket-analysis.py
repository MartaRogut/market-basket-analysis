import os
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = "basket_analysis1.csv"
MIN_SUPPORT = 0.05
MIN_LIFT = 1.0
MIN_CONFIDENCE_STRONG = 0.5

TOP_N_ITEMS = 10
TOP_N_RULES = 10
TOP_N_3D = 100

SAVE_PLOTS = False
OUTPUT_DIR = "outputs"

def ensure_output_dir():
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)


    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.astype(bool)

    return df


def print_basic_info(df: pd.DataFrame) -> None:
    print("Podgląd danych:")
    print(df.head())
    print("Liczba transakcji:", len(df))
    print("Liczba produktów (kolumn):", df.shape[1])


def mine_frequent_itemsets_apriori(df: pd.DataFrame) -> pd.DataFrame:
    frequent = apriori(df, min_support=MIN_SUPPORT, use_colnames=True)
    return frequent.sort_values("support", ascending=False)

def mine_frequent_itemsets_fpgrowth(df: pd.DataFrame) -> pd.DataFrame:
    frequent = fpgrowth(df, min_support=MIN_SUPPORT, use_colnames=True)
    return frequent.sort_values("support", ascending=False)

def mine_rules(frequent_itemsets: pd.DataFrame) -> pd.DataFrame:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=MIN_LIFT)
    return rules.sort_values(by="lift", ascending=False)

def format_itemset(itemset) -> str:
    # itemset jest frozenset
    return ", ".join(sorted(list(itemset)))

def plot_top_items_support(frequent_itemsets: pd.DataFrame) -> None:
    top_items = frequent_itemsets.head(TOP_N_ITEMS).copy()
    top_items["product_name"] = top_items["itemsets"].apply(format_itemset)

    plt.figure(figsize=(10, 5))
    plt.bar(top_items["product_name"], top_items["support"])
    for i, v in enumerate(top_items["support"]):
        plt.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=10)

    plt.title(f"{TOP_N_ITEMS} najczęściej występujących itemsetów wg support (Apriori)", fontsize=14, fontweight="bold")
    plt.xlabel("Produkt / zestaw produktów", fontsize=12)
    plt.ylabel("Support", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if SAVE_PLOTS:
        ensure_output_dir()
        plt.savefig(os.path.join(OUTPUT_DIR, "top_items_support_apriori.png"), dpi=200)

    plt.show()
def plot_top_rules_lift(rules: pd.DataFrame) -> None:
    top_rules = rules.head(TOP_N_RULES).copy()

    rule_labels = [
        f"{format_itemset(row['antecedents'])} ⇒ {format_itemset(row['consequents'])}"
        for _, row in top_rules.iterrows()
    ]

    plt.figure(figsize=(12, 6))
    plt.bar(rule_labels, top_rules["lift"])
    for i, v in enumerate(top_rules["lift"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    plt.title(f"{TOP_N_RULES} reguł asocjacyjnych wg lift (Apriori)", fontsize=14, fontweight="bold")
    plt.xlabel("Reguła asocjacyjna", fontsize=12)
    plt.ylabel("Lift", fontsize=12)
    plt.xticks(rotation=60, ha="right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if SAVE_PLOTS:
        ensure_output_dir()
        plt.savefig(os.path.join(OUTPUT_DIR, "top_rules_lift_apriori.png"), dpi=200)

    plt.show()


def interpret_best_rule(rules: pd.DataFrame) -> None:
    if rules.empty:
        print("\nBrak reguł do interpretacji.")
        return

    best_rule = rules.iloc[0]
    ante = format_itemset(best_rule["antecedents"])
    cons = format_itemset(best_rule["consequents"])
    sup = best_rule["support"]
    conf = best_rule["confidence"]
    lift = best_rule["lift"]

    print("\nPrzykładowa reguła – interpretacja:")
    print(f"Reguła: jeśli klient kupi {{{ante}}}, to zwykle kupuje też {{{cons}}}.")
    print(f"Support = {sup:.3f} → odsetek transakcji zawierających jednocześnie: {{{ante}}} i {{{cons}}}.")
    print(f"Confidence = {conf:.3f} → odsetek przypadków, gdy kupiono {{{ante}}}, a dokupiono też {{{cons}}}.")
    print(f"Lift = {lift:.3f} → ile razy częściej {{{cons}}} pojawia się z {{{ante}}} niż przy niezależnych zakupach.")


def plot_rules_3d(rules: pd.DataFrame) -> None:
    if rules.empty:
        return

    rules_3d = rules.head(TOP_N_3D).copy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        rules_3d["support"],
        rules_3d["confidence"],
        rules_3d["lift"],
        c=rules_3d["lift"],
        cmap="viridis",
        s=40,
    )

    ax.set_title("Reguły asocjacyjne – 3D (support, confidence, lift)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Support", fontsize=10)
    ax.set_ylabel("Confidence", fontsize=10)
    ax.set_zlabel("Lift", fontsize=10)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("Lift")

    plt.tight_layout()

    if SAVE_PLOTS:
        ensure_output_dir()
        plt.savefig(os.path.join(OUTPUT_DIR, "rules_3d_apriori.png"), dpi=200)

    plt.show()

def compare_algorithms(frequent_apriori: pd.DataFrame, frequent_fp: pd.DataFrame,
                       rules_apriori: pd.DataFrame, rules_fp: pd.DataFrame) -> None:
    top_apriori = frequent_apriori.head(TOP_N_ITEMS)["itemsets"].apply(format_itemset)
    top_fp = frequent_fp.head(TOP_N_ITEMS)["itemsets"].apply(format_itemset)

    comparison = pd.DataFrame({
        "Apriori": top_apriori.values,
        "FP-Growth": top_fp.values
    })

    print("\nPorównanie algorytmów Apriori vs FP-Growth (Top itemsets):")
    print(comparison)

    print(f"\nLiczba reguł – Apriori: {len(rules_apriori)}")
    print(f"Liczba reguł – FP-Growth: {len(rules_fp)}")

    strong_ap = rules_apriori[rules_apriori["confidence"] > MIN_CONFIDENCE_STRONG]
    strong_fp = rules_fp[rules_fp["confidence"] > MIN_CONFIDENCE_STRONG]
    print(f"Liczba reguł z confidence > {MIN_CONFIDENCE_STRONG} – Apriori: {len(strong_ap)}")
    print(f"Liczba reguł z confidence > {MIN_CONFIDENCE_STRONG} – FP-Growth: {len(strong_fp)}")

def main():
    df = load_data(DATA_PATH)
    print_basic_info(df)

    # APRIORI
    frequent_apriori = mine_frequent_itemsets_apriori(df)
    print("\nNajczęściej kupowane produkty / itemsety (Apriori):")
    print(frequent_apriori.head(TOP_N_ITEMS))

    rules_apriori = mine_rules(frequent_apriori)
    print("\nNajsilniejsze reguły asocjacyjne (Apriori) – top 10 wg lift:")
    if not rules_apriori.empty:
        print(rules_apriori[["antecedents", "consequents", "support", "confidence", "lift"]].head(TOP_N_RULES))
    else:
        print("Brak reguł (spróbuj zmniejszyć min_support lub min_lift).")

    # FP-GROWTH
    frequent_fp = mine_frequent_itemsets_fpgrowth(df)
    print("\nNajczęściej kupowane produkty / itemsety (FP-Growth):")
    print(frequent_fp.head(TOP_N_ITEMS))

    rules_fp = mine_rules(frequent_fp)
    print("\nNajsilniejsze reguły asocjacyjne (FP-Growth) – top 10 wg lift:")
    if not rules_fp.empty:
        print(rules_fp[["antecedents", "consequents", "support", "confidence", "lift"]].head(TOP_N_RULES))
    else:
        print("Brak reguł (spróbuj zmniejszyć min_support lub min_lift).")

    plot_top_items_support(frequent_apriori)
    plot_top_rules_lift(rules_apriori)
    interpret_best_rule(rules_apriori)
    plot_rules_3d(rules_apriori)

    compare_algorithms(frequent_apriori, frequent_fp, rules_apriori, rules_fp)


if __name__ == "__main__":
    main()
