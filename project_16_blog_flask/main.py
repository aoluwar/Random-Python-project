# Flask blog scaffold. To run: install Flask and run this file.
from flask import Flask, render_template_string
app = Flask(__name__)
POSTS = [{'title':'Hello','slug':'hello','body':'This is a post.'}]
@app.route('/')
def index():
    return render_template_string('<h1>Blog</h1>{% for p in posts %}<h2>{{p.title}}</h2>{% endfor %}', posts=POSTS)
if __name__=='__main__': app.run(debug=True)
