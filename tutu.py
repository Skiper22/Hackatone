from flask import Flask, render_template, request
from bokeh.plotting import figure
from bokeh.embed import components
import numpy as np
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import CustomJS, Button
from bokeh.layouts import row
from pmdarima import auto_arima
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import joblib
from datetime import datetime

def pred_ARIMA(aer1, aer2, i, data):
    model_name = f'data/model_ARIMA_{aer1}-{aer2}_{i}.joblib'
    l_index_name = f'data/last_date_index_ARIMA_{aer1}-{aer2}_{i}.txt'
    model = joblib.load(model_name)
    with open(l_index_name, 'r') as file:
        last_index = file.read()
    last_index = datetime.strptime(last_index, "%Y-%m-%d %H:%M:%S")
    new_index = datetime.strptime(data, "%Y-%m-%d")
    if new_index < last_index:
        return 'error'
    else:
        len_pred = 30


        intermediate_dates = pd.date_range(start=last_index, end=new_index + pd.DateOffset(days=len_pred), freq='D')
        new_data = pd.DataFrame({'SDAT_S': [last_index] + intermediate_dates.tolist()})


        predictions = model.predict(n_periods=len(new_data), X=new_data)

        predictions = np.square(predictions)
        predictions = np.maximum(predictions, 0)
        predictions = np.round(predictions)

        return [list(new_data[-len_pred:]['SDAT_S'].values), list(predictions[-len_pred:])]

# def pred_PROPHET(aer1, aer2, i, data):
    # model_name = f'data/model_PROPHET_{aer1}-{aer2}_{i}.joblib'
    # l_index_name = f'data/last_date_index_PROPHET_{aer1}-{aer2}_{i}.txt'
    # model = joblib.load(model_name)
    # with open(l_index_name, 'r') as file:
    #     last_index = file.read()
    # last_index = datetime.strptime(last_index, "%Y-%m-%d %H:%M:%S")
    # data = datetime.strptime(data, "%Y-%m-%d")
    # new_index = datetime.strptime(data, "%d-%m-%Y")
    # if data < last_index:
    #     return 'error'
    # else:
    #
    #     future = pd.date_range(start=new_index, periods=len_pred + 1, freq='D')[:-1]
    #     future = pd.DataFrame(future, columns=['ds'])
    #
    #     predictions = model.predict(future)
    #
    #     predictions = np.square(predictions['yhat'])
    #     predictions = predictions.values
    #
    #     predictions = np.maximum(predictions, 0)
    #     predictions = np.round(predictions)

        # return ()

app = Flask(__name__)


def print_graphic(x ,y):

    # Заполнение массива рандомными значениями
    x = [i for i in range(1, 31)]

    y = [int(i) for i in y]
    # Вычисляем начальный и конечный индексы для отображения 20 точек
    start_index = 0
    end_index = 10

    # Вычисляем минимальное и максимальное значение по оси x
    x_min = 1
    x_max = 30

    # Вычисляем минимальное и максимальное значение по оси y
    y_min = min(y[start_index:end_index])
    y_max = max(y[start_index:end_index])

    # output to static HTML file
    output_file("lines.html")

    # create a new plot with a title and axis labels
    p = figure(
        title="Определение динамики бронирований",
        x_axis_label='даты',
        y_axis_label='Предсказанное значение',
        width=1200,  # Set the plot width to 1200 pixels
        height=400,  # Set the plot height to 400 pixels
        x_range=(x_min, x_max),
        y_range=(y_min, y_max)  # Set the y-axis range based on data
    )

    # add a line renderer with legend and line thickness
    p.line(x, y, legend_label="Динамика", line_width=2)

    # Calculate the histogram data
    # hist, edges = np.histogram(y, bins='auto')

    # Add the histogram as a quad glyph
    # p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.5)

    # Make the histogram visible and the line invisible
    # p.renderers = [p.renderers[-1]]  # Keep only the last renderer (histogram)

    # Create buttons for scrolling the plot
    scroll_left_button = Button(label='←', width=50)
    scroll_right_button = Button(label='→', width=50)

    # Define the JavaScript callback function for scrolling left
    scroll_left_callback = CustomJS(args=dict(x_range=p.x_range), code="""
        x_range.start -= 1;
        x_range.end -= 1;
    """)

    # Define the JavaScript callback function for scrolling right
    scroll_right_callback = CustomJS(args=dict(x_range=p.x_range), code="""
        x_range.start += 1;
        x_range.end += 1;
    """)

    # Attach the callback functions to the buttons
    scroll_left_button.js_on_click(scroll_left_callback)
    scroll_right_button.js_on_click(scroll_right_callback)

    # Arrange the plot and buttons in a layout
    layout = row(p, scroll_left_button, scroll_right_button)

    # show the results
    show(layout)
    return p

@app.route('/')
def index():
    return render_template('file.html')


@app.route('/process-form', methods=['POST'])
def process_form():
    flight_direction = request.form['flight-direction']
    flight_class = request.form['flight-class']
    flight_date = request.form['flight-date']

    # Вывод выбранных значений в Python
    # print("Выбрано направление: ", flight_direction[:3],flight_direction[4:])
    # print("Выбран класс: ", flight_class)
    print("Выбрана Дата:", type(flight_date), flight_date)
    # Вызов мат моделей

    get_values = pred_ARIMA(flight_direction[:3], flight_direction[4:], flight_class, flight_date)
    if get_values != "error":# список дат для оси Х и список значений для У
        p = print_graphic(get_values[0], get_values[1])
        print()

    # get_values = pred_PROPHET(flight_direction[:3], flight_direction[4:], flight_class, flight_date)
    # if get_values != "error":  # список дат для оси Х и список значений для У
    #     print_graphic(get_values[0], get_values[1])

    script, div = components(p)
    # print_graphic()
    return render_template('result.html', script=script, div=div)


if __name__ == '__main__':
    app.run(port=5009)
#02.12.2019
