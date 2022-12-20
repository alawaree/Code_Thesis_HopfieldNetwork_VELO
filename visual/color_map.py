from colorsys import hsv_to_rgb


class Colormap:

    # Defines a colormap in HSV spectrum adapting H for each color
    def __init__(self, min_v, max_v, min_c, max_c, sat=1, val=1):
        self.min_v = min_v
        self.min_c = min_c
        self.span_v = max_v - min_v
        self.span_c = max_c - min_c
        self.sat = sat
        self.val = val

    def get_color_rgb(self, v):
        c = self.get_color_hsv(v)
        return hsv_to_rgb(c[0], c[1], c[2])

    def get_color_hsv(self, v):
        return (self.min_c + ((v - self.min_v) / self.span_v) * self.span_c,
                self.sat, self.val)
