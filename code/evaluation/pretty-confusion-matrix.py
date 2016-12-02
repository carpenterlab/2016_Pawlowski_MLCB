#!/usr/bin/env python

import sys
from optparse import OptionParser
import numpy as np
import cairo
from cpf.profiling.leave_one_out import confusion_matrix
from cpf.profiling.confusion import load_confusion

def normal_font(ctx):
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_font_size(7)
    ctx.select_font_face('Helvetica')

def bold_font(ctx):
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_font_size(7)
    ctx.select_font_face('Helvetica', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)

def small_font(ctx):
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_font_size(7)
    ctx.select_font_face('Helvetica')


class Title(object):
    def __init__(self, title):
        self.title = title

    def set_font(self, ctx):
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_font_size(7)
        ctx.select_font_face('Helvetica', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)

    def height(self, ctx):
        if self.title:
            ctx.save()
            self.set_font(ctx)
            h = ctx.text_extents(self.title)[3]
            ctx.restore()
            return h + 5
        else:
            return 0

    def draw(self, ctx):
        if self.title:
            self.set_font(ctx)
            h = self.height(ctx) - 5
            ctx.move_to(0, h)
            ctx.show_text(self.title)
            ctx.stroke()


class Header(object):
    strings = ['True mechanistic class', 'Predicted class', 'Acc.']

    def set_font(self, ctx):
        normal_font(ctx)

    def height(self, ctx):
        ctx.save()
        self.set_font(ctx)
        h = max([ctx.text_extents(s)[3] for s in self.strings])
        ctx.restore()
        return h + 5

    def draw(self, ctx, matrix_left, matrix_size, figure_width):
        self.set_font(ctx)
        h = self.height(ctx) - 5
        ctx.move_to(0, h)
        ctx.show_text(self.strings[0])

        w = ctx.text_extents(self.strings[1])[4]
        x2c = matrix_left + 0.5 * matrix_size
        ctx.move_to(x2c - w / 2.0, h)
        ctx.show_text(self.strings[1])

        w = ctx.text_extents(self.strings[2])[4]
        ctx.move_to(figure_width - w, h)
        ctx.show_text(self.strings[2])

        ctx.stroke()


class Subheader(object):
    def __init__(self, codes):
        self.codes = codes

    def set_font(self, ctx):
        small_font(ctx)

    def height(self, ctx):
        ctx.save()
        self.set_font(ctx)
        h = max([ctx.text_extents(s)[3] for s in self.codes])
        ctx.restore()
        return h + 5

    def draw(self, ctx, matrix_left, matrix_width):
        self.set_font(ctx)
        h = self.height(ctx) - 5
        tile_size = matrix_width / len(self.codes)
        for i, code in enumerate(self.codes):
            w = ctx.text_extents(code)[3]
            ctx.move_to(matrix_left + (i + 0.5) * tile_size + w / 2.0, h + 3) # - w / 2.0, h)
            ctx.save()
            ctx.rotate(3 * 3.14/2.0)
            ctx.show_text(code)
            ctx.restore()
        ctx.stroke()

class Body(object):
    def __init__(self, figure_width, labels, cm, formatter=None):
        self.figure_width = figure_width
        self.labels = labels
        self.cm = cm
        self.accuracies = ['%.0f %%' % (cm[i,i] * 100.0 / cm[i,:].sum())
                           for i in range(len(self.labels))]
        if formatter is None:
            self.formatter = lambda i, j: cm[i, j]
        else:
            self.formatter = formatter

    def set_font(self, ctx):
        normal_font(ctx)

    def set_inner_font(self, ctx):
        small_font(ctx)

    def code_left(self, ctx):
        self.set_font(ctx)
        class_width = max([ctx.text_extents(s)[4] for s, code in self.labels])
        return class_width + 5

    def matrix_left(self, ctx):
        self.set_font(ctx)
        code_width = max([ctx.text_extents(code)[4] for s, code in self.labels])
        return self.code_left(ctx) + code_width + 5

    def matrix_size(self, ctx):
        ctx.save()
        normal_font(ctx)
        acc_width = ctx.text_extents('100 %')[4]
        matrix_size = self.figure_width - self.matrix_left(ctx) - acc_width - 5
        ctx.restore()
        return matrix_size

    def matrix_width(self, ctx):
        return self.matrix_size(ctx)

    def height(self, ctx):
        return self.matrix_size(ctx) + 5

    def draw(self, ctx):
        lw = 0.5
        inner_size = self.matrix_size(ctx) - 2 * lw
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(lw)
        ml = self.matrix_left(ctx)
        ctx.rectangle(ml + lw, lw, inner_size, inner_size)
        ctx.stroke()
        n = len(self.labels)
        tile_size = inner_size / n
        self.set_inner_font(ctx)
        for i in range(n):
            for j in range(n):
                ctx.rectangle(ml + lw + j * tile_size, lw + i * tile_size,
                              tile_size, tile_size)
                ctx.set_source_rgba(1, 0, 0, self.cm[i, j] * 0.8 / self.cm[i,:].sum())
                ctx.fill()
                # Number inside
                ctx.set_source_rgba(0, 0, 0)
                if self.cm.dtype.kind == 'f':
                    if self.cm[i, j] < 0.5:
                        s = ''
                    #elif self.cm[i, j] < 0.95:
                    #    s = '.%d' % round(self.cm[i, j] * 10.0)
                    else:
                        s = '%d' % round(self.cm[i, j])
                elif self.cm.dtype.kind == 'i':
                    if self.cm[i, j] == 0:
                        s = ''
                    else:
                        s = '%d' % self.cm[i, j]
                else:
                    s = str(self.cm[i, k])
                h, w = ctx.text_extents(s)[3:5]
                ctx.move_to(ml + lw + (j + 0.5) * tile_size - 0.5 * w,
                            lw + (i + 0.5) * tile_size + 0.5 * h)
                ctx.show_text(s)
                ctx.stroke()

        ctx.set_source_rgb(0, 0, 0) # black

        self.set_font(ctx)
        cl = self.code_left(ctx)
        for i, (s, code) in enumerate(self.labels):
            h = ctx.text_extents(s)[3]
            y = (i + 0.5) * tile_size + h / 2.0
            ctx.move_to(0, y)
            ctx.show_text(s)
            ctx.move_to(cl, y)
            ctx.show_text(code)

        for i, acc in enumerate(self.accuracies):
            h, w = ctx.text_extents(acc)[3:5]
            y = (i + 0.5) * tile_size + h / 2.0
            ctx.move_to(self.figure_width - w, y)
            ctx.show_text(acc)
        ctx.stroke()


class Footer(object):
    def __init__(self, figure_width, cm):
        self.figure_width = figure_width
        self.cm = cm
        self.ncorrect = sum(self.cm[i,i] for i in range(self.cm.shape[0]))
        self.ntotal = self.cm.sum()

    def left_font(self, ctx):
        normal_font(ctx)

    def left_text(self):
        if self.cm.dtype.kind == 'f':
            return 'Overall accuracy: '
        else:
            return 'Overall accuracy: %d / %d = ' % (self.ncorrect, self.ntotal)

    def right_font(self, ctx):
        bold_font(ctx)

    def right_text(self):
        return '%.0f %%' % (self.ncorrect * 100.0 / self.ntotal)

    def height(self, ctx):
        ctx.save()
        self.left_font(ctx)
        h1 = ctx.text_extents(self.left_text())[3]
        self.right_font(ctx)
        h2 = ctx.text_extents(self.right_text())[3]
        ctx.restore()
        return max(h1, h2)

    def draw(self, ctx):
        y = self.height(ctx) - 3

        self.right_font(ctx)
        s = self.right_text()
        wr = ctx.text_extents(s)[4]
        ctx.move_to(self.figure_width - wr, y)
        ctx.show_text(s)

        self.left_font(ctx)
        s = self.left_text()
        wl = ctx.text_extents(s)[4]
        ctx.move_to(self.figure_width - wr - wl, y)
        ctx.show_text(s)

        ctx.stroke()


class Figure(object):
    def __init__(self, figure_width, labels, cm, title=None):
        self.figure_width = figure_width
        self.title = Title(title)
        self.header = Header()
        codes = [code for s, code in labels]
        self.subheader = Subheader(codes)
        self.body = Body(figure_width, labels, cm)
        self.footer = Footer(figure_width, cm)

    def height(self, ctx):
        return (self.title.height(ctx) + self.header.height(ctx) +
                self.subheader.height(ctx) +
                self.body.height(ctx) + self.footer.height(ctx))

    def draw(self, ctx):
        matrix_left = self.body.matrix_left(ctx)
        matrix_size = self.body.matrix_size(ctx)
        self.title.draw(ctx)
        ctx.translate(0, self.title.height(ctx))
        self.header.draw(ctx, matrix_left, matrix_size, self.figure_width)
        ctx.translate(0, self.header.height(ctx))
        self.subheader.draw(ctx, matrix_left, matrix_size)
        ctx.translate(0, self.subheader.height(ctx))
        self.body.draw(ctx)
        ctx.translate(0, self.body.height(ctx))
        self.footer.draw(ctx)

class Page(object):
    def __init__(self, figure, margin=0):
        self.figure = figure
        self.margin = margin

    def height(self, ctx):
        return self.figure.height(ctx) + 2 * self.margin

    def draw(self, ctx):
        ctx.save()
        ctx.translate(margin, margin)
        self.figure.draw(ctx)
        ctx.restore()


code_map = dict([('Actin disruptors', 'Act'),
                 ('Aurora kinase inhibitors', 'Aur'),
                 ('Cholesterol-lowering', 'Ch'),
                 ('DNA damage', 'DD'),
                 ('DNA replication', 'DR'),
                 ('Epithelial', 'Epi'),
                 ('Kinase inhibitors', 'KI'),
                 ('Monoaster', 'MA'),
                 ('Eg5 inhibitors', 'Eg5'),
                 ('Microtubule stabilizers', 'MS'),
                 ('Microtubule destabilizers', 'MD'),
                 ('Protein degradation', 'PD'),
                 ('Protein synthesis', 'PS'),
                 # Loo et al.
                 ('Actin', 'Act'),
                 ('Calcium regulation', 'Ca'),
                 ('Cholesterol', 'Cho'),
                 ('Cyclooxygenase', 'Cyc'),
                 ('Energy metabolism', 'En'),
                 ('Histone deacetylase', 'HD'),
                 ('Kinase', 'Kin'),
                 ('Microtubule', 'MT'),
                 ('Neurotransmitter', 'Ne'),
                 ('Nuclear receptor', 'Nu'),
                 ('Topoisomerase', 'Topo'),
                 ('Vesicle trafficing', 'Ves'),
                 ('Metal homeostasis', 'MH')])

parser = OptionParser("usage: %prog [-f] [-t TITLE] INPUT-FILE OUTPUT-FILE")
parser.add_option('-f', dest='float', action='store_true', help='use floating-point accuracies')
parser.add_option('-t', dest='title', help='title')
options, args = parser.parse_args()
if len(args) != 2:
    parser.error('Incorrect number of arguments')
input_filename, output_filename = args

confusion = load_confusion(input_filename)
cm = confusion_matrix(confusion, 'if'[options.float or 0])
labels = [(l, code_map.get(l, l))
          for l in sorted(set(a for a, b in confusion.keys()))]

figure_width = 3.44 * 72
figure = Figure(figure_width, labels, cm, options.title)
margin = 0
page = Page(figure, margin)

surface_width = figure_width + 2 * margin
surface_height = 675.36 + 2 * margin
surface = cairo.PDFSurface(output_filename, surface_width, surface_height)
ctx = cairo.Context(surface)
surface.set_size(surface_width, page.height(ctx))
page.draw(ctx)
surface.show_page()
surface.finish()
