from PIL import Image
import pytesseract

img = Image.open("/home/ivan/Desktop/Diplomska/teseract_test/инјекции.jpeg")


text = pytesseract.image_to_string(img, lang='mkd')
print(text)

