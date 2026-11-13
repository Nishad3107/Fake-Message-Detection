"""Load persisted model artifacts and produce a prediction."""

from pathlib import Path

import joblib

from app.ml.model_utils import clean_text, model_paths


class ModelPredictor:
    def __init__(self, model_dir: Path) -> None:
        classifier_path, vectorizer_path = model_paths(model_dir)
        self.classifier = joblib.load(classifier_path)
        self.vectorizer = joblib.load(vectorizer_path)
        try:
            self.classifier.predict_proba(self.vectorizer.transform([""]))
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                "Model artifact was created by an incompatible scikit-learn version"
            ) from exc

    def predict(self, text: str) -> tuple[str, float, list[str]]:
        vectors = self.vectorizer.transform([clean_text(text)])
        probabilities = self.classifier.predict_proba(vectors)[0]
        index = int(probabilities.argmax())
        label = str(self.classifier.classes_[index])
        terms: list[str] = []
        if hasattr(self.classifier, "coef_"):
            coefficients = self.classifier.coef_[0]
            positive_label = str(self.classifier.classes_[1])
            contributions = vectors.toarray()[0] * coefficients
            if label != positive_label:
                contributions = -contributions
            feature_names = self.vectorizer.get_feature_names_out()
            terms = [
                str(feature_names[i])
                for i in contributions.argsort()[::-1]
                if contributions[i] > 0
            ][:5]
        return label, float(probabilities[index]), terms
