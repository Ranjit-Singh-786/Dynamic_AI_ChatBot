from flask import Flask, render_template, request

app = Flask(__name__)

# Example function to simulate chatbot responses
def get_chatbot_response(user_message):
    # You'll replace this with your actual NLP model's logic
    bot_response = "This is a sample bot response."
    return bot_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['user_message']
    bot_response = get_chatbot_response(user_message)
    return bot_response

if __name__ == '__main__':
    app.run(debug=True)
