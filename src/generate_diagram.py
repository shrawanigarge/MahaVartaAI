"""
Generate Architecture Diagram for Marathi News Classification Project
"""

import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram():
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)

    # Helper function for rounded boxes
    def draw_box(x, y, w, h, text, color="#2563EB", text_color="white", fontsize=10, bold=True):
        box = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.8,rounding_size=2",
            facecolor=color, edgecolor="#1E293B", linewidth=1.5,
            transform=ax.transData
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight=weight, multialignment="center")

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#1E293B", lw=2, mutation_scale=15)
        )

    # Title
    ax.text(50, 115, "Marathi News Classification System Architecture", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#0F172A")

    # Nodes
    # 1. Headline
    draw_box(50, 106, 60, 6, "Marathi News Headline\n(e.g. Hrithik Biopic 'Super 30')", color="#1E293B")
    draw_arrow(50, 103, 50, 96)

    # 2. Preprocessing
    draw_box(50, 93, 55, 6, "Indic Text Preprocessing\n(Unicode NFKC Normalization & Cleaning)", color="#0284C7")
    draw_arrow(50, 90, 50, 83)

    # 3. Tokenization
    draw_box(50, 80, 50, 5, "Marathi Tokenization", color="#0D9488")
    draw_arrow(50, 77.5, 50, 71.5)

    # 4. Stopwords
    draw_box(50, 69, 50, 5, "Marathi Stopword Removal", color="#059669")
    draw_arrow(50, 66.5, 50, 60)

    # 5. TF-IDF
    draw_box(50, 57, 50, 6, "TF-IDF Feature Extraction\n(Unigram + Bigram, 6,884 Features)", color="#7C3AED")

    # Branching arrows to 3 models
    draw_arrow(50, 54, 20, 47)
    draw_arrow(50, 54, 50, 47)
    draw_arrow(50, 54, 80, 47)

    # Models
    draw_box(20, 44, 26, 6, "Multinomial\nNaive Bayes (82.80%)", color="#D97706")
    draw_box(50, 44, 26, 6, "Logistic\nRegression (82.76%)", color="#D97706")
    draw_box(80, 44, 26, 6, "Linear SVM\nCalibrated (83.75%)", color="#059669")

    # Convergence arrows
    draw_arrow(20, 41, 50, 34)
    draw_arrow(50, 41, 50, 34)
    draw_arrow(80, 41, 50, 34)

    # Best Classifier
    draw_box(50, 31, 50, 6, "Optimal Model Selection\nLinear SVM (Validation F1: 83.20%)", color="#2563EB")
    draw_arrow(50, 28, 50, 22)

    # Predicted Category
    draw_box(50, 19, 55, 6, "Predicted News Category & Confidence\n(State • Entertainment • Sports)", color="#DB2777")
    draw_arrow(50, 16, 50, 10.5)

    # Deployment / App
    draw_box(50, 7.5, 60, 6, "Flask Web Application & REST API (/api/predict)\nBrowser UI • JSON API", color="#0F172A")

    plt.tight_layout()
    output_path = Path("results") / "architecture_diagram.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    # Copy to static/images
    static_path = Path("static") / "images" / "architecture_diagram.png"
    static_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(output_path, static_path)
    print(f"Architecture diagram saved to {output_path} and {static_path}")

    # Text version
    text_diagram = """
                    Marathi News Headline
                              │
                              ▼
                     Text Preprocessing
                              │
                              ▼
                    Marathi Tokenization
                              │
                              ▼
                      Stopword Removal
                              │
                              ▼
                           TF-IDF
               (Unigrams + Bigrams, Sublinear TF)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Naive Bayes    Logistic Regression  SVM
          (Acc: 82.80%)     (Acc: 82.76%)  (Acc: 83.75%)
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                       Best Classifier
                     Linear SVM (83.20% F1)
                              │
                              ▼
                    Predicted News Category
                 (State • Entertainment • Sports)
                              │
                              ▼
                       Flask Web App
                              │
                              ▼
                       User / REST API
"""
    with open(Path("results") / "architecture_diagram.txt", "w", encoding="utf-8") as f:
        f.write(text_diagram)
    print("Architecture text diagram saved to results/architecture_diagram.txt")

if __name__ == "__main__":
    generate_diagram()
