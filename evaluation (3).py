from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#predict
ypred = model.predict(xtest)
ypred = le.inverse_transform(ypred)
ypred = pd.DataFrame(ypred)

guess_home_team = pd.DataFrame(['H']*76)

print("Accuracy:", accuracy_score(ytest, ypred))
print("Accuracy:", accuracy_score(ytest, guess_home_team))
print(confusion_matrix(ytest, ypred))
print(classification_report(ytest, ypred))
print(combined_stats)


