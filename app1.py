from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64


app = Flask(__name__)
model = pickle.load(open('model3.pkl', 'rb'))
dataset = pd.read_csv('processed.csv')

@app.route("/")
def hello():
    return render_template("index2.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    mxv=[38.0,30.4,9.01,3350.0,10.1,5.8,3949.0]
    l=0
    if request.method == "POST":
        input_features=[]
        for x in request.form.values():
            input_features.append(float(x)/mxv[l])
            l=l+1
        features_value = [np.array(input_features)]

        feature_names = ['Temp', 'D.O. (mg/l)', 'PH', 'CONDUCTIVITY (µmhos/cm)', 'B.O.D. (mg/l)',
       'NITRATENAN N+ NITRITENANN (mg/l)', 'TOTAL COLIFORM (MPN/100ml)Mean']

        df = pd.DataFrame(features_value, columns = feature_names)
        output = model.predict(features_value)
        if output==5:
            p="Excellent"
        elif output==4:
           p="Good"
        elif output==3:
            p="Okay"
        elif output==2:
            p="Poor"
        elif output==1:
            p="Very poor"  
         
    return render_template('predpage.html',qua=p)

@app.route("/predpage")
def predpage():
   predict()


@app.route("/index")
def index():
     return render_template('index.html')  
 

@app.route("/index")
def generate_plot(temperature):
    # Filter dataset based on temperature
    filtered_data = dataset[dataset['Temp'] == temperature]
    
    # Count occurrences of each value in X column
    j = filtered_data['wqc'].value_counts().sort_index()
    
    m=j.sum()
    s=[]
    z=1
    if len(j)==0:
        plot_url=0
        return plot_url 
    for i in j.index:
     r=((j[i]/m)*100)
     b=round(r,2)
     s.append(b)
    z=0
    for i in j.index:
     j[i]=s[z]
     z=z+1# Plotting
     
     
    plt.figure(figsize=(8, 6))
    j.plot(kind='bar',)
    plt.xticks(rotation=0,horizontalalignment="center")
    plt.xlabel('water quality class')
    plt.ylabel('Percentage')
    plt.title('Bar plot for {} Temperature'.format(temperature))
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/index/p', methods=['GET', 'POST'])
def p():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        plot_url = generate_plot(temperature)
        return render_template('index.html', plot_url=plot_url)
    return render_template('index.html')
  
@app.route("/ph")
def ph():
    return render_template('ph.html')  
 
@app.route("/ph")
def generate_plotph(phgraph):
    # Filter dataset based on temperature
    filtered_data = dataset[dataset['PH'] == phgraph]
    
    # Count occurrences of each value in X column
    j = filtered_data['wqc'].value_counts().sort_index()
    
    
    m=j.sum()
    s=[]
    z=1
    if len(j)==0:
        plot_url=0
        return plot_url
    for i in j.index:
     r=((j[i]/m)*100)
     b=round(r,2)
     s.append(b)
    z=0
    for i in j.index:
     j[i]=s[z]
     z=z+1# Plotting
     

    plt.figure(figsize=(8, 6))
    j.plot(kind='bar')
    plt.xticks(rotation=0,horizontalalignment="center")
    plt.xlabel('water quality class')
    plt.ylabel('Percentage')
    plt.title('Bar plot for {} PH'.format(phgraph))
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/ph/phin', methods=['GET', 'POST'])
def phin():
    if request.method == 'POST':
        phgraph = float(request.form['phval'])
        plot_url = generate_plotph(phgraph)
        return render_template('ph.html', plot_url=plot_url)
    return render_template('ph.html')

@app.route("/tc")
def tc():
    return render_template('tc.html')  
 
@app.route("/tc")
def generate_plottc(tcgraph):
    # Filter dataset based on temperature
    filtered_data = dataset[dataset['TOTAL COLIFORM (MPN/100ml)Mean'] == tcgraph]
    
    # Count occurrences of each value in X column
    j = filtered_data['wqc'].value_counts().sort_index()
    
    m=j.sum()
    s=[]
    z=1
    if len(j)==0:
        plot_url=0
        return plot_url 
    for i in j.index:
     r=((j[i]/m)*100)
     b=round(r,2)
     s.append(b)
    z=0
    for i in j.index:
     j[i]=s[z]
     z=z+1# Plotting
     
    plt.figure(figsize=(8, 6))
    j.plot(kind='bar')
    plt.xticks(rotation=0,horizontalalignment="center")
    plt.xlabel('water quality class')
    plt.ylabel('Percentage')
    plt.title('Bar plot for {} Total coliform'.format(tcgraph))
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/tc/tcin', methods=['GET', 'POST'])
def tcin():
    if request.method == 'POST':
        tcgraph = float(request.form['tcval'])
        plot_url = generate_plottc(tcgraph)
        return render_template('tc.html', plot_url=plot_url)
    return render_template('tc.html')

@app.route("/nitrate")
def nitrate():
    return render_template('nitrate.html')  

@app.route("/nitrate")
def plotnitrate(nitrategraph):
    # Filter dataset based on nitrate
    filtered_data = dataset[dataset['NITRATENAN N+ NITRITENANN (mg/l)'] == nitrategraph]
    
    # Count occurrences of each value in X column
    j = filtered_data['wqc'].value_counts().sort_index()
    
    m=j.sum()
    s=[]
    z=1
    if len(j)==0:
        plot_url=0
        return plot_url 
    for i in j.index:
        r=((j[i]/m)*100)
        b=round(r,2)
        s.append(b)
    z=0
    for i in j.index:
        j[i]=s[z]
        z=z+1# Plotting
     
    plt.figure(figsize=(8, 6))
    j.plot(kind='bar')
    plt.xticks(rotation=0,horizontalalignment="center")
    plt.xlabel('water quality class')
    plt.ylabel('Percentage')
    plt.title('Bar plot for {} Nitrate'.format(nitrategraph))
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/nitrate/nitratein', methods=['GET', 'POST'])
def nitratein():
    if request.method == 'POST':
        nitrategraph = float(request.form['nitrateval'])
        plot_url = plotnitrate(nitrategraph)
        return render_template('nitrate.html', plot_url=plot_url)
    return render_template('nitrate.html')

@app.route("/bod")
def bod():
    return render_template('bod.html')  

@app.route("/bod")
def plotbod(bodgraph):
    # Filter dataset based on BOD
    filtered_data = dataset[dataset['B.O.D. (mg/l)'] == bodgraph]
    
    # Count occurrences of each value in X column
    j = filtered_data['wqc'].value_counts().sort_index()
    
    m=j.sum()
    s=[]
    z=1
    if len(j)==0:
        plot_url=0
        return plot_url
    for i in j.index:
        r=((j[i]/m)*100)
        b=round(r,2)
        s.append(b)
    z=0
    for i in j.index:
        j[i]=s[z]
        z=z+1# Plotting
     
    plt.figure(figsize=(8, 6))
    j.plot(kind='bar')
    plt.xticks(rotation=0,horizontalalignment="center")
    plt.xlabel('water quality class')
    plt.ylabel('Percentage')
    plt.title('Bar plot for {} BOD'.format(bodgraph))
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/bod/bodin', methods=['GET', 'POST'])
def bodin():
    if request.method == 'POST':
        bodgraph = float(request.form['bodval'])
        plot_url = plotbod(bodgraph)
        return render_template('bod.html', plot_url=plot_url)
    return render_template('bod.html')

@app.route("/conductivity")
def conductivity():
    return render_template('conductivity.html')  

@app.route("/conductivity")
def plotcnd(cndgraph):
    # Filter dataset based on Conductivity
    filtered_data = dataset[dataset['CONDUCTIVITY (umhos/cm)'] == cndgraph]
    
    # Count occurrences of each value in X column
    j = filtered_data['wqc'].value_counts().sort_index()
    
    m=j.sum()
    s=[]
    z=1
    if len(j)==0:
        plot_url=0
        return plot_url
    for i in j.index:
        r=((j[i]/m)*100)
        b=round(r,2)
        s.append(b)
    z=0
    for i in j.index:
        j[i]=s[z]
        z=z+1# Plotting
     
    plt.figure(figsize=(8, 6))
    j.plot(kind='bar')
    plt.xticks(rotation=0,horizontalalignment="center")
    plt.xlabel('water quality class')
    plt.ylabel('Percentage')
    plt.title('Bar plot for {} Conductivity'.format(cndgraph))
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/conductivity/conductivityin', methods=['GET', 'POST'])
def conductivityin():
    if request.method == 'POST':
        cndgraph = float(request.form['conductivityval'])
        plot_url = plotcnd(cndgraph)
        return render_template('conductivity.html', plot_url=plot_url)
    return render_template('conductivity.html')

@app.route("/do")
def do():
    return render_template('do.html')  

@app.route("/do")
def plotdo(dograph):
    # Filter dataset based on Dissolved Oxygen
    filtered_data = dataset[dataset['D.O. (mg/l)'] == dograph]
    
    # Count occurrences of each value in X column
    j = filtered_data['wqc'].value_counts().sort_index()
    
    
    m = j.sum()
    s = []
    z = 1
    if len(j)==0:
        plot_url=0
        return plot_url
    for i in j.index:
        r = ((j[i] / m) * 100)
        b = round(r, 2)
        s.append(b)
    z = 0
    for i in j.index:
        j[i] = s[z]
        z = z + 1  # Plotting
     
    plt.figure(figsize=(8, 6))
    j.plot(kind='bar')
    plt.xticks(rotation=0,horizontalalignment="center")
    plt.xlabel('water quality class')
    plt.ylabel('Percentage')
    plt.title('Bar plot for {} Dissolved Oxygen'.format(dograph))
    # Save plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/do/doin', methods=['GET', 'POST'])
def doin():
    if request.method == 'POST':
        dograph = float(request.form['doval'])
        plot_url = plotdo(dograph)
        return render_template('do.html', plot_url=plot_url)
    return render_template('do.html')

if __name__ == "__main__":
    app.run(debug=True)
