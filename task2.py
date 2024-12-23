import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("SDN_traffic.csv")
data.info()

X = data.drop(columns=["category", "id_flow", "nw_src", "nw_dst"])
X["forward_bps_var"] = X["forward_bps_var"].str.replace(r"[^\d+eE\-.]", "", regex=True).astype(float)
y = data["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ID3
clf_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_id3.fit(X_train, y_train)
y_pred_id3 = clf_id3.predict(X_test)

print("\n=== ID3 ===")
print("Accuracy score:", accuracy_score(y_test, y_pred_id3))
print(classification_report(y_test, y_pred_id3))

# CART
clf_cart = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_cart.fit(X_train, y_train)
y_pred_cart = clf_cart.predict(X_test)

print("\n=== CART ===")
print("Accuracy score:", accuracy_score(y_test, y_pred_cart))
print(classification_report(y_test, y_pred_cart))

print("\nComparison:")
print(f"ID3 accuracy: {accuracy_score(y_test, y_pred_id3):.4f}")
print(f"CART accuracy: {accuracy_score(y_test, y_pred_cart):.4f}")
