from collections import defaultdict

from PIL import Image, ImageDraw
from neuron import Network, Layer
import numpy as np


class DrawNet:

    LAYER_WIDTH = 128
    LAYER_HEIGHT = 256
    MARGIN = 64

    def __init__(self, network):
        self.layers = network.layers[1:]

        self.draw_data = []

        self.layer_connecting_points = defaultdict(list)
        self.total_weights = 0

    def _draw(self, style, *args, **kwargs):
        self.draw_data.append((style, args, kwargs))

    def _draw_layer(self, layer):
        which = layer.order

        if which > len(self.layers):
            raise ValueError("layer number two large")

        self._draw_layer_outline(layer)
        self._draw_layer_name(layer)
        self._draw_weights(layer)

    def _draw_weights(self, layer):

        for n, (l, d) in enumerate(layer.InputLayers):
            onpoint, offpoint = self._get_input_points(layer, n)

            self._draw("line", (offpoint, onpoint), fill=0)
            self._draw_arrow(onpoint, 'R')
            self._draw_weightext(onpoint, layer.Ws[(l, d)], d)

            self.layer_connecting_points[l].append((layer, offpoint, d))
            self.total_weights += 1

    def _draw_weights_connections(self, layer):
        for n, (l, p, d) in enumerate(self.layer_connecting_points[layer]):

            on, off, bridge = self._get_output_points(layer, l, d, p, n+1)

            self._draw("line", (on, off), fill=0)
            self._draw("line", (off, bridge), fill=0)
            self._draw("line", (bridge, p), fill=0)

    def _get_output_points(self, outlayer, inlayer, D, P, n):
        if outlayer.order + 1 == inlayer.order:
            on = (self._rightx(outlayer), P[1])
            off = on
            bridge = P
            return on, off, bridge

        else:
            step_dx = len(self.layer_connecting_points[outlayer]) + 1
            step_x = self.LAYER_WIDTH / step_dx

            step_dy = sum(len(v) for v in self.layer_connecting_points.values())
            step_y = self.MARGIN / step_dy

            if outlayer.order < inlayer.order:
                on = (self._leftx(outlayer) + n*step_x,
                    self._upy(outlayer))

                off = (self._leftx(outlayer) + n*step_x,
                    self._upy(outlayer) - (outlayer.order)*n*step_y)
            else:
                on = (self._leftx(outlayer) + n*step_x,
                    self._downy(outlayer))

                off = (self._leftx(outlayer) + n*step_x,
                    self._downy(outlayer) + (outlayer.order)*n*step_y)

            bridge = (P[0], off[1])

            return on, off, bridge

    def _get_input_points(self, layer, n):
        step = (self.MARGIN // (len(layer.InputLayers) + 1),
                self.LAYER_HEIGHT // (len(layer.InputLayers) + 1))

        on = (self._leftx(layer),
              self._downy(layer) - (n + 1)*step[1])
        off = (self._leftx(layer) - (n + 1)*step[0],
               self._downy(layer) - (n + 1)*step[1])

        return (on, off)

    def _draw_weightext(self, point, W, D):
        weight_text = "W(D={0})[{1}]".format(D, W.shape)
        self._draw("text", (point[0] + self.MARGIN//10, point[1]),
                              weight_text, fill=0)

    def _draw_arrow(self, point, direction):
        if direction == "right" or "R":
            dir_t = (-1, -1, -1, 1)

        arrow_point = (point[0] + dir_t[0]*self.MARGIN//10,
                        point[1] + dir_t[1]*self.MARGIN//10)
        self._draw("line", (arrow_point, point), fill=0)

        arrow_point = (point[0] + dir_t[2]*self.MARGIN//10,
                        point[1] + dir_t[3]*self.MARGIN//10)
        self._draw("line", (arrow_point, point), fill=0)

    def _draw_layer_outline(self, layer):
        start = (self._leftx(layer), self._upy(layer))
        end = (self._rightx(layer), self._downy(layer))

        self._draw("rectangle", (start, end), outline=0)

    def _draw_layer_name(self, layer):
        which = layer.order
        size = layer.size
        layer_name = "Layer {0}, N={1}".format(which, size)
        text_start = (self._leftx(layer), self._upy(layer))

        self._draw("text", text_start, layer_name, fill=0)

    def _draw_input(self):
        layer = self.layers[0]
        end = (self._leftx(layer), self._upy(layer) + self.MARGIN/2)
        start = (end[0] - self.MARGIN, end[1])
        self._draw('line', (start, end), fill=0)
        self._draw_arrow(end, 'right')

    def _draw_output(self):
        layer = self.layers[-1]
        start = (self._rightx(layer), self._upy(layer) + self.MARGIN/2)
        end = (start[0] + self.MARGIN, start[1])
        self._draw('line', (start, end), fill=0)
        self._draw_arrow(end, 'right')

    def _leftx(self, layer):
        which = layer.order
        return self.MARGIN + which*(self.LAYER_WIDTH + self.MARGIN)

    def _rightx(self, layer):
        return self._leftx(layer) + self.LAYER_WIDTH

    def _upy(self, layer):
        return self.MARGIN

    def _downy(self, layer):
        return self._upy(layer) + self.LAYER_HEIGHT

    def _draw_net(self):
        for l in self.layers:
            self._draw_layer(l)

        for l in self.layers:
            self._draw_weights_connections(l)

        self._draw_input()
        self._draw_output()

    def draw(self):
        self._draw_net()

    def show(self):
        num_of_layers = len(self.layers) + 2 #plus input and output

        size = ((self.LAYER_WIDTH + self.MARGIN)*num_of_layers + 2*self.MARGIN,
                self.LAYER_HEIGHT + 3*self.MARGIN)

        img = Image.new("RGBA", size, (255, 255, 255, 0))
        imgdraw = ImageDraw.Draw(img)

        for style, args, kwargs in self.draw_data:
            getattr(imgdraw, style)(*args, **kwargs)

        img.show()

if __name__ == "__main__":

    l1 = Layer(2)
    l2 = Layer(1, B=np.array([0]))
    l3 = Layer(3, B=np.array([0]))

    l1.A = np.array([[1, 1], [2, 1]])

    n = Network((l1, l2, l3))
    n.connect(1, 2, D=0)
    n.connect(2, 1, D=1)
    n.connect(2, 3, D=0)
    n.connect(3, 2, D=1)
    n.connect(3, 1, D=2)

    c = DrawNet(n)
    c.draw()
    print(c.layer_connecting_points)
    c.show()
