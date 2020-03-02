from tkinter import *
from tkinter.colorchooser import askcolor
import PIL
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess) 


class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.image1 = PIL.Image.new('RGB', (600, 600), 'white')
        self.draw = ImageDraw.Draw(self.image1)

        self.clear_canvas = Button(self.root, text='CLEAR', command=self.clear_canvas)
        self.clear_canvas.grid(row=0, column=5)

        self.setup()
        self.load_model()
        self.first_prediction = True
        self.root.mainloop()

    def clear_canvas(self):
        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=1, columnspan=5)
        self.image1 = PIL.Image.new('RGB', (600, 600), 'white')
        self.draw = ImageDraw.Draw(self.image1)
        self.setup()

    def load_model(self):
        with open('models/fruits7.json', 'r') as file:
            loaded_json = file.read()
        # clean json:
        loaded_json = loaded_json.replace(", \"interpolation\": \"nearest\"", '')
        loaded_json = loaded_json.replace(", \"output_padding\": null", '')
        loaded_json = loaded_json.replace(", \"ragged\": false", '')
        loaded_json = loaded_json.replace(", \"kernel_initializer\": {\"class_name\": \"GlorotUniform\", \"config\": {\"seed\": null, \"dtype\": \"float32\"}}", '')
        self.loaded_model = model_from_json(loaded_json)
        self.loaded_model.load_weights('models/fruits7.h5')
        self.loaded_model.compile(loss='mse', optimizer='adam')

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill=paint_color, width=self.line_width)
            #self.predict()
        self.old_x = event.x
        self.old_y = event.y
	
    def predict(self):
        resized = cv2.resize(np.array(self.image1, dtype='float'), (200,200))[:,:,0]/255.0
        resized[resized > 0.5] = 1.0
        resized[resized <= 0.5] = 0.0
        preprocessed = 1 - resized
        prediction = self.loaded_model.predict(preprocessed.reshape(1,200,200,1)).reshape(200,200,3)
        prediction = prediction/np.max(prediction)
        sketch = cv2.resize(np.array(self.image1, dtype='float'), (200,200))
        sketch = sketch/np.max(sketch)
        merge = prediction + (sketch*0.2)
        print(sketch)
        print(prediction)
        if self.first_prediction:
            plotting = plt.imshow(prediction)
            plt.axis('off')
            plt.show()
            self.first_prediction = False
        else:
            plotting.set_data(prediction)
            plt.draw()

    def reset(self, event):
        self.old_x, self.old_y = None, None
        self.predict()


if __name__ == '__main__':
    Paint()