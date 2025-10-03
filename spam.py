import re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

CSV = "k_shalamberidze2024_481659.csv"
FEATS = ["words","links","capital_words","spam_word_count"]
LABEL = "is_spam"

# 1) Load
df = pd.read_csv(CSV)

# 2) Visuals 
df[LABEL].value_counts().sort_index().plot(kind="bar", title="Class balance (0=legit,1=spam)")
plt.tight_layout(); plt.savefig("class_balance.png", dpi=150); plt.close()

plt.imshow(df[FEATS].corr(), interpolation="nearest"); plt.title("Feature correlation")
plt.xticks(range(len(FEATS)), FEATS, rotation=45, ha="right"); plt.yticks(range(len(FEATS)), FEATS)
plt.colorbar(); plt.tight_layout(); plt.savefig("feature_correlation.png", dpi=150); plt.close()

# 3) Split 70/30
X = df[FEATS].to_numpy(float); y = df[LABEL].to_numpy(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# 4) Standardize + Train LR
scaler = StandardScaler().fit(Xtr)
clf = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42).fit(scaler.transform(Xtr), ytr)

# 5) Test
yp = clf.predict(scaler.transform(Xte))
acc = accuracy_score(yte, yp); cm = confusion_matrix(yte, yp)

# 6) Save CM fig + print summary for report
plt.imshow(cm); plt.title("Confusion Matrix"); plt.xticks([0,1],["Legit","Spam"]); plt.yticks([0,1],["Legit","Spam"])
for i in range(2):
    for j in range(2): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

print("Accuracy:", round(acc,4))
print("Intercept:", round(clf.intercept_[0],4))
for f,c in zip(FEATS, clf.coef_.ravel()): print(f"{f}: {c:+.4f}")

# 7) Email-text checker (same features)
SPAM_WORDS = {"free","offer","win","winner","money","credit","urgent","deal","click","bonus","prize","limited","now","guarantee","lottery"}

def extract(text:str):
    words = re.findall(r"\b\w+\b", text)
    return {
        "words": len(words),
        "links": len(re.findall(r"https?://|www\.", text, flags=re.I)),
        "capital_words": sum(1 for w in words if w.isupper() and len(w)>1),
        "spam_word_count": sum(len(re.findall(rf"\b{w}\b", text, flags=re.I)) for w in SPAM_WORDS),
    }

def predict_text(text:str):
    v = np.array([[extract(text)[k] for k in FEATS]], float)
    p = clf.predict_proba(scaler.transform(v))[0,1]
    print("\nText:", text); print("Features:", extract(text)); print("P(spam):", round(p,3), "=>", "spam" if p>=0.5 else "legit")

# two examples to include in report
predict_text("URGENT! You WIN a $5000 PRIZE now! Click https://win.example for your FREE offer!!!")
predict_text("Hello team, attaching notes from today’s lecture. No links. Thanks.")
