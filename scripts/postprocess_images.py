from PIL import Image, ImageDraw, ImageFont

VISDRONE_CLASS_MARKUPS = [
    {"id": 0, "label": "IGNORED", "color": (0, 0, 0)},
    {"id": 1, "label": "PEDESTRIAN", "color": (255, 153, 0)},
    {"id": 2, "label": "PEOPLE", "color": (255, 102, 0)},
    {"id": 3, "label": "BICYCLE", "color": (0, 255, 0)},
    {"id": 4, "label": "CAR", "color": (255, 255, 255)},
    {"id": 5, "label": "VAN", "color": (255, 0, 255)},
    {"id": 6, "label": "TRUCK", "color": (255, 0, 0)},
    {"id": 7, "label": "TRICYCLE", "color": (0, 0, 255)},
    {"id": 8, "label": "TRI-AWN", "color": (0, 0, 128)},
    {"id": 9, "label": "BUS", "color": (255, 255, 0)},
    {"id": 10, "label": "MOTOR", "color": (153, 153, 255)},
    {"id": 11, "label": "OTHERS", "color": (153, 51, 102)},
]

TTF_FONT_PATH = "./fonts/Roboto-Black.ttf"


def paint_visdrone_labels(img_filename, labels, class_markups, thresh=0.5):
    """
    Creates an image with a bounding box overlay for detected objects based on the VisDrone class labels
        Parameters:
          img_filename (str): filepath to source image
          labels (list): list of numpy arrays for each visdrone class label
                         each entry in the numpy array is structured as: 
                             (upperleft_y, upperleft_x, lowerright_y, lowerright_x, confidence)
          class_markups (list): list of class label visualization dicts. Each entry has the following
                                keys: id, label, color
          thresh (float): cutoff threshold for predicted label confidence. Predictions with values less than
                          thresh will be ignored.
        Returns:
          Returns a new RGB image with class label overlay
    """

    font = ImageFont.truetype(TTF_FONT_PATH, 15)
    img_base = Image.open(img_filename).convert("RGBA")
    img_overlay_area = Image.new("RGBA", img_base.size)
    img_overlay_box = Image.new("RGBA", img_base.size)
    draw = ImageDraw.Draw(img_overlay_area)
    boxdraw = ImageDraw.Draw(img_overlay_box)
    for markup in class_markups:
        for entry in labels[markup["id"]]:
            if entry[-1] > thresh:
                draw.rectangle(
                    [(entry[0], entry[1]), (entry[2], entry[3])],
                    outline=markup["color"],
                    fill=markup["color"] + (128,),
                )
                boxdraw.rectangle(
                    [(entry[0], entry[1]), (entry[2], entry[3])],
                    outline=markup["color"],
                )
    draw.rectangle([(10, 10), (135, 230)], outline="black", fill=(255, 255, 255, 128))
    for markup in class_markups:
        draw.rectangle(
            [(15, 15 + markup["id"] * 18), (30, 30 + markup["id"] * 18)],
            outline="black",
            fill=markup["color"],
        )
        draw.text((35, 15 + markup["id"] * 18), markup["label"], "black", font=font)

    img = Image.alpha_composite(img_base, img_overlay_area)
    img = Image.alpha_composite(img, img_overlay_box)
    img = img.convert("RGB")
    return img
