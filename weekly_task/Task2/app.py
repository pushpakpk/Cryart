from flask import Flask, request, render_template_string
from nlp_pipeline import preprocess_text

app = Flask(__name__)

# Simple HTML template
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Task 2 - NLP Preprocessing</title>
</head>
<body style="font-family: Arial; margin: 30px;">
    <h2>Text Preprocessing with NLTK</h2>
    <form method="POST">
        <textarea name="text" rows="6" cols="60" placeholder="Enter text here...">{{ text }}</textarea><br><br>
        <button type="submit">Preprocess</button>
    </form>
    {% if tokens %}
        <h3>Processed Tokens:</h3>
        <p>{{ tokens }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    tokens, text = None, ""
    if request.method == "POST":
        text = request.form["text"]
        tokens = preprocess_text(text)
    return render_template_string(html_template, tokens=tokens, text=text)

if __name__ == "__main__":
    app.run(debug=True)
