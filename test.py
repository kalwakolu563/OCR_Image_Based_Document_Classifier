import pytesseract
# from PIL import Image
from PIL import Image
# print(Image.__version__)  # Should print Pillow version without errors



# Load the image
image = Image.open("C:\\Users\\Admin\\Pictures\\Screenshots\\Screenshot 2025-05-20 210823.png")
# image = Image.open("C:\\Users\\Admin\\OneDrive - Predictive Research Inc\\Pictures\\Screenshot 2025-05-22 051346.png")

# Extract text
text = pytesseract.image_to_string(image)

print("Extracted Text:")
print(text)



