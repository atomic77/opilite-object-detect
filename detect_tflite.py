import argparse
import io
import time
import numpy as np

from PIL import Image, ImageColor, ImageDraw, ImageFont,  ImageOps

from tflite_runtime.interpreter import Interpreter

# Module level vars
interpreter = None
labels = None

def load_labels(path):
    with open(path) as f:
        return {
            int(s.split("  ")[0]): s.split("  ")[1].strip()
            for s in f.readlines()
        }

def initialize(model_file='detect.tflite', labels_file='coco_labels.txt'):
    global interpreter , labels
    interpreter = Interpreter(model_file)
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    interpreter.allocate_tensors()
    labels = load_labels(labels_file)


def set_input_tensor(image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(image, threshold, classes_incl=None):
    """Returns a list of detection results, each a dictionary of object info. If classes is None, return all"""
    set_input_tensor(image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(0)
    classes = get_output_tensor(1)
    scores = get_output_tensor(2)
    count = int(get_output_tensor(3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': int(classes[i]),
                'score': scores[i]
            }
            if not classes_incl:
                results.append(result)
            elif classes[i] in classes_incl:
                results.append(result)
    return results


def obj_detect_from_pil(img, threshold=0.3, classes_incl=None):
    if not interpreter:
        initialize()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    image = img.convert('RGB').resize((width, height), Image.ANTIALIAS)
    results = detect_objects(image, threshold, classes_incl)
    return results


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=2, display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_box(image, boxes, class_name, score, max_boxes=10, min_score=0.1):
    """Simplified from obj detection full TF example"""
    colors = list(ImageColor.colormap.values())
    font = ImageFont.load_default()

    ymin, xmin, ymax, xmax = tuple(boxes)
    display_str = "{}: {}%".format(class_name, int(100 * score))
    color = colors[hash(class_name) % len(colors)]
    draw_bounding_box_on_image(
        image,ymin,xmin,ymax,xmax,color,font,display_str_list=[display_str]
    )

def draw_boxes(image, results, min_score=0.2, max_boxes=10):
    """ Draw boxes from results structure onto image """
    results = sorted(results, key=lambda x: x['score'])
    results = results[0:max_boxes]
    for r in results:
        if r['score'] < min_score:
            continue
        draw_box(image, r['bounding_box'], labels[r['class_id']], r['score'])