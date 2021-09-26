from flask import Flask,render_template,request
import pickle
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

app= Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])

def predict():

    if request.method == 'POST':

        cr = request.form['cr']
        ra = request.form['ra']
        aq = request.form['aq']
        rn = request.form['rn']
        age = request.form['age']
        teh = request.form['teh']
        pp = request.form['pp']
        nbed = request.form['nbed']
        nbat = request.form['nbat']
        rf = request.form['rf']
        ad = request.form['ad']
        ap = request.form['ap']
        wl = request.form['wl']
        lr = request.form['lr']
        wr = request.form['wr']

        price = np.array([[float(cr),float(ra),float(aq),float(rn),float(age),float(teh),float(pp),float(nbed),float(nbat),int(rf),float(ad),np.uint8(ap),np.uint8(wl),np.uint8(lr),np.uint8(wr)]])
        
        model = pickle.load(open('hpp.pkl','rb'))
        prediction = model.predict(price)

    return render_template('index.html', prediction = str(prediction))

if __name__=='__main__':
    app.run()