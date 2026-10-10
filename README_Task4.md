# YouTube Comment Sentiment Classifier (Marvel's Wolverine Launch)

This project is a narrow AI system: a machine learning classifier built
for one task — sentiment classification of YouTube comments — with a
live demo, documented failures, and ethics notes.

Video analyzed: "Wolverine slashes his way onto @spherevegas to celebrate
the launch of Marvel's Wolverine"
(https://www.youtube.com/watch?v=ZvbifV4mKb0)

## 1. Problem
Classify YouTube comments on PlayStation's Marvel's Wolverine launch
video as **Positive**, **Negative**, or **Neutral**, so audience reaction
can be read at a glance instead of scrolling through every comment.

**User:** Community managers and game marketers who want a quick, rough
read on audience sentiment.

## 2. Method

### Data
- 100 public YouTube comments collected with the `youtube-comment-downloader`
  library.
- Labeled by hand; 16 spam/unreadable comments removed, leaving
  **83 labeled comments**.
- Only comment text was kept — no usernames or personal data.
- Labeling rules: Positive = excitement/praise, Negative =
  complaints/sarcasm, Neutral = questions/facts.
- Label counts: Negative 57, Neutral 20, Positive 6

### Model
- TF-IDF (unigrams + bigrams) + Logistic Regression (`class_weight='balanced'`)
- 80/20 stratified train/test split (66 train, 17 test)
- Trained model saved with `joblib` (`sentiment_model.pkl`) and reloaded
  for reuse, no retraining needed per prediction

### Integration
```python
loaded_model = joblib.load('sentiment_model.pkl')

def predict_sentiment(comment):
    if not comment or not isinstance(comment, str) or not comment.strip():
        return "Error: input must be a non-empty string"
    prediction = loaded_model.predict([comment])
    return prediction[0]
```
The function validates input before calling the model, so empty strings,
whitespace, `None`, and non-string types return a clear error instead of
crashing.

### Constraints
- Free tools only (Google Colab, scikit-learn, joblib, ipywidgets)
- No external API or secret keys
- Prediction takes under 1 second per comment

## 3. Demo

### Live interface (notebook)
A simple `ipywidgets` text box + button lets anyone type a comment and
get an instant prediction:

```python
import ipywidgets as widgets
from IPython.display import display

text_box = widgets.Text(placeholder='Type a comment...')
button = widgets.Button(description='Predict')
output = widgets.Output()

def on_click(b):
    output.clear_output()
    with output:
        print(predict_sentiment(text_box.value))

button.on_click(on_click)
display(text_box, button, output)
```
Example: typing `"Not bad, not great either"` and clicking **Predict**
returns `Negative`.

### Evaluation examples
| Input | Expected | Predicted | Result |
|---|---|---|---|
| 'This trailer looks incredible!' | Positive | Positive | ✅ |
| 'Bring back physical games!' | Negative | Negative | ✅ |
| '5/10' | Neutral | Neutral | ✅ |

### Edge cases
| Input | Predicted | Note |
|---|---|---|
| 'ok' | Negative | Too short/ambiguous |
| 'yeah right, amazing launch' | Negative | Sarcasm, correct by coincidence |
| 'SONY SONY SONY' | Negative | No sentiment words, unreliable |
| '🤩🤩🤩' | Negative | Emoji-only, not understood |
| 'Not bad, not great either' | Negative | Mixed sentiment, oversimplified |

### Failure cases (from real test data)
| Comment | Actual | Predicted | Likely Reason |
|---|---|---|---|
| "'play has no limits'. If you follow the straight line that it does have limits." | Positive | Negative | Figurative language |
| "recent decisions...have been awful. Thank you for destroying the once beloved brand." | Neutral | Negative | Mixed tone, strong words dominate |
| "I bought the physical version of this game! Great job Insomniac! Shame on you Sony!" | Negative | Neutral | Praise + criticism combined confuses model |
| "GREED HAS NO LIMITS!" | Negative | Neutral | All-caps emotional tone not captured |

**Pattern:** most failures involve sarcasm, mixed sentiment within one
comment, or emotional tone without explicit keywords — known limitations
of TF-IDF + Logistic Regression, which has no real understanding of
context or tone.

## 4. Results Summary
- Accuracy: 58.8%
- Macro F1: 0.32
- Per-class F1: Negative 0.72, Neutral 0.25, Positive 0.00

**Success criteria:** Target was 70% accuracy / 0.60 macro F1. Current
results are below target, mainly due to the small, imbalanced dataset
(only 6 Positive examples).

## 5. Limitations
- Small, heavily imbalanced dataset (83 comments, mostly Negative)
- Positive-class predictions are unreliable (only 6 training examples)
- No handling of sarcasm, irony, or mixed sentiment within one comment
- Data comes from a single video, so results may not generalize to other
  content or communities
- English-language comments only

## 6. Ethics Notes
- **Privacy:** only public comment text was collected; no usernames,
  profile data, or other personal identifiers were stored.
- **Bias:** the dataset is skewed toward Negative comments, so the model
  is more likely to mislabel Positive or Neutral text as Negative. This
  is a dataset bias, not a judgment about the commenters.
- **Not for moderation or decisions:** given the low accuracy, this model
  should not be used to moderate, flag, or make decisions about real
  users. It is a learning prototype, not a production tool.
- **Transparency:** failure cases and accuracy are reported honestly here
  rather than hidden, so anyone using this project understands its limits.
- **No sensitive data:** no health, biometric, financial, or otherwise
  sensitive personal data was used or inferred.

## 7. Future Work
- Collect more comments, especially Positive examples, to balance the
  dataset
- Use 5-fold cross-validation for more reliable metrics on small data
- Try a pretrained transformer (e.g., DistilBERT) for better handling of
  sarcasm and context
- Add a confidence score to flag low-confidence predictions for review

## Files
- `comments_labeled.csv`: labeled dataset
- `sentiment_model.pkl`: saved trained model
- `notebook.ipynb`: full pipeline — collection, labeling, training,
  evaluation, failure analysis, and live demo
- `README.md`: this document
