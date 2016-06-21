import datetime
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/cnn/<sicd>')
def process(sicd) :
    return render_template('index.html', sicd=sicd,
               datetime=datetime.date.today().strftime('%Y%m%d%H%M%S'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')