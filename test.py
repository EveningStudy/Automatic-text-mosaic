import mosaicType
import cv2

if __name__ == '__main__':
    image_path = '222.png'
    image = cv2.imread(image_path)
    padding = 5
    lang = ['ch_sim', 'en']
    image = mosaicType.common(image, padding,lang)
    cv2.imwrite('111_ma.png', image)
