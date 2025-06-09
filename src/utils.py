from paddleocr import TextDetection

model = TextDetection(model_name="PP-OCRv5_server_det", enable_mkldnn=False)

def detect_boxes(img_path, filename):
    result = model.predict(img_path)
    for row in result:
        row.save_to_img(f"output/detected/{filename}.jpg")
        
        return row