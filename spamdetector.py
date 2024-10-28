import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = [
    ("Exclusive deal just for you! Get a free trial for our premium service.", "spam"),
    ("Reminder: Your appointment is scheduled for tomorrow at 10 AM.", "ham"),
    ("Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize!", "spam"),
    ("Just checking in to see how you're doing. Let me know if you need anything.", "ham"),
    ("Limited time offer: Buy 2 get 1 free on all products! Don't miss out!", "spam"),
    ("Hi! Are we still on for lunch next week?", "ham"),
    ("Your account has been compromised. Please verify your identity to prevent closure.", "spam"),
    ("Thanks for your help with the project! I really appreciate it.", "ham"),
    ("Unlock your chance to earn big with our new investment opportunity. Sign up today!", "spam"),
    ("Can you send me the updated report by EOD?", "ham"),
    ("Get paid to take surveys! It's easy money from the comfort of your home!", "spam"),
    ("Don't forget to bring your laptop to the meeting tomorrow.", "ham"),
    ("Urgent: Update your payment information to avoid service interruption.", "spam"),
    ("Looking forward to our vacation together! Let me know your flight details.", "ham"),
    ("Act now! Limited stocks available for our exclusive membership offer.", "spam"),
    ("Have you watched the latest episode of our favorite series? Let's discuss!", "ham"),
    ("Earn $500 a day working from home! Sign up now!", "spam"),
    ("Just a friendly reminder about the webinar this Friday at 2 PM.", "ham"),
    ("Your subscription will be automatically renewed unless you cancel. Act now!", "spam"),
    ("It was great catching up last weekend! Let's plan to do it again soon.", "ham"),
    ("Join our weight loss program and shed those pounds quickly!", "spam"),
    ("I sent you the files you requested. Please check your email.", "ham"),
    ("Congratulations! You have been selected for an all-expenses-paid trip!", "spam"),
    ("Hope you're having a great week! Just wanted to touch base on our project.", "ham"),
    ("Your invoice is ready. Click here to view and pay your bill.", "spam"),
    ("Can we schedule a time to discuss the new proposal?", "ham"),
    ("Get rich quick! Invest in cryptocurrency today!", "spam"),
    ("Let's meet for coffee next week. How does Wednesday sound?", "ham"),
    ("You have won a free vacation! Call now to claim your prize.", "spam"),
    ("Great job on the presentation! It was well received.", "ham"),
    ("Surprise! You're eligible for a special discount on your next purchase.", "spam"),
    ("Have a wonderful weekend! Looking forward to our next meeting.", "ham"),
    ("Act now to get a free gift card! Limited time only!", "spam"),
    ("Lets finalize the details for the project by the end of the day.", "ham"),
    ("Free trial of our premium membership available! Sign up now!", "spam"),
    ("I appreciate your prompt response. It really helps move things along.", "ham"),
    ("New diet plan guarantees results! Try it today!", "spam"),
    ("Just wanted to confirm our plans for the weekend.", "ham"),
    ("Claim your free iPhone now! Supplies are limited!", "spam"),
    ("Thanks for your support on the recent changes. It means a lot!", "ham")
]
df = pd.DataFrame(data, columns=['message', 'label'])

X_train = df['message']
y_train = df['label']

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

while True:
    test = input("Enter a test case: ")
    if test.lower() == 'stop':
        break
    test_vec = vectorizer.transform([test])
    pred = model.predict(test_vec)
    print("The machine predicted the given test as: ",pred[0])
    answer = input("What was the corect answer?: ").strip().lower()
    if(answer != pred[0]):
        data.append((test,answer))
        df = pd.DataFrame(data, columns=['message', 'label'])
        X_train = df['message']
        y_train = df['label']
        X_train_vec = vectorizer.fit_transform(X_train)
        model.fit(X_train_vec, y_train)
        print("Model updated with the new example.")
    else:
        print("The prediction was corect!")