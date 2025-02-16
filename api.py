from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Load trained models & vectorizer
nb_model = joblib.load('naive_bayes_model.pkl')
log_reg_model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = FastAPI()

class InputData(BaseModel):
    text: str

@app.post("/predict/")
def predict(data: InputData, model_type: str = "naive_bayes"):
    # Convert input text to TF-IDF
    transformed_text = vectorizer.transform([data.text])
    
    # Choose model
    if model_type == "naive_bayes":
        prediction = nb_model.predict(transformed_text)[0]
    elif model_type == "logistic_regression":
        prediction = log_reg_model.predict(transformed_text)[0]
    else:
        return {"error": "Invalid model type. Choose 'naive_bayes' or 'logistic_regression'."}
    
    return {"predicted_specialty": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
