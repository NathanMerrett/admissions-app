from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model = None

@app.before_first_request
def load_model():
    global model
    model = tf.keras.models.load_model("my_model.h5")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from form and preprocess it
        data = preprocess_data(request.form)
        
        # Make prediction
        prediction = model.predict(data)
        
        return jsonify({'prediction': prediction.tolist()})
    
    return render_template('index.html')