from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('app.html')


@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
@app.route('/get_home_types', methods=['GET'])
def get_home_types():
    response = jsonify({
        'home_types': util.get_home_types()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    area_sqft = float(request.form['area_sqft'])
    location = request.form['location']
    property_type = request.form['hometype']
    bedrooms = int(request.form['bedrooms'])
    baths = int(request.form['baths'])

    # response = jsonify({
    #     'estimated_price': util.get_estimated_price(location,area_sqft,bedrooms,baths,property_type)
    # })
    # response.headers.add('Access-Control-Allow-Origin', '*')
    # print(response)
    data = int(util.get_estimated_price(location,area_sqft,bedrooms,baths,property_type))
    # print(data)

    return render_template('app.html',data=data)

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    util.load_saved_artifacts()
    app.run(debug=True)