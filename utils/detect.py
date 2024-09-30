import easyocr


def detectBox(image, lang=['ch_sim', 'en'], gpu=True):
    reader = easyocr.Reader(lang, gpu)
    h_list, v_list = reader.detect(image)
    h_list, v_list = h_list[0], v_list[0]
    return h_list, v_list
