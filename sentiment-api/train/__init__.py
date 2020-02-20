import pickle


class Sentiment:
    def __init__(self):
        self.tfidf = pickle.load(open("tfidf.pkl", "rb"))
        self.model = pickle.load(open("model.pkl", "rb"))

    def predict(self, text):
        text_predict = self.tfidf.transform([text])
        return self.model.predict(text_predict)
