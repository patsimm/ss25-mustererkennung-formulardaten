from utils import detect_boxes, match_boxes, visualize_matches, crop_image_to_rows

if __name__ == "__main__":
    img_path = "/Users/mdo/Documents/HTWK/6. Semester/Mustererkennung/ss25-mustererkennung-formulardaten/src/img_handwritten.jpg"
    print("Erkenne Labels...")
    _, label_texts, _, label_centers, label_heights = detect_boxes(img_path, "detected_boxes")