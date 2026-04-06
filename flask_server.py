from flask import Flask, request, redirect, render_template_string
import requests
import datetime

app = Flask(__name__)

PI_URL = "http://172.20.10.6:5000"

results = []
last_timestamp = None

def clean_text(text):
    parts = text.split("-")
    seen = set()
    clean = []

    for p in parts:
        p = p.strip()
        if p and p not in seen:
            clean.append(p)
            seen.add(p)

    return "\n• " + "\n• ".join(clean)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Pi Vision Dashboard</title>

<style>
body {
    background: #0f172a;
    color: white;
    font-family: Arial;
    text-align: center;
}

button {
    padding: 15px 30px;
    font-size: 18px;
    background: #22c55e;
    border: none;
    border-radius: 10px;
    cursor: pointer;
}

.card {
    background: #1e293b;
    margin: 20px auto;
    padding: 20px;
    width: 70%;
    border-radius: 12px;
    text-align: left;
}

.time {
    color: #38bdf8;
    font-weight: bold;
}

img {
    width: 300px;
    max-width: 100%;
    display: block;
    margin: 10px auto;
    border-radius: 10px;
}

.status {
    color: orange;
}
</style>

<script>
setTimeout(() => location.reload(), 3000);
</script>

</head>
<body>

<h1>📡 Pi Vision Dashboard</h1>

<form method="POST" action="/trigger">
    <button>Run Inference</button>
</form>

{% if running %}
<p class="status">⏳ Processing...</p>
{% endif %}

<hr>

{% for r in results %}
<div class="card">
    <div class="time">{{ r.time }}</div>
    <img src="data:image/jpeg;base64,{{ r.image }}">
    <pre>{{ r.result }}</pre>
</div>
{% endfor %}

</body>
</html>
"""

@app.route("/")
def index():
    global last_timestamp

    try:
        res = requests.get(f"{PI_URL}/result", timeout=2)
        data = res.json()

        if data["result"] and data["timestamp"] != last_timestamp:
            results.append({
                "time": data["timestamp"],
                "result": clean_text(data["result"]),
                "image": data["image"]
            })
            last_timestamp = data["timestamp"]

        running = data.get("running", False)

    except:
        running = False

    return render_template_string(HTML, results=results, running=running)

@app.route("/trigger", methods=["POST"])
def trigger():
    try:
        requests.post(f"{PI_URL}/trigger", timeout=5)
    except:
        pass

    return redirect("/")

if __name__ == "__main__":
    app.run(port=5000, debug=True) 
