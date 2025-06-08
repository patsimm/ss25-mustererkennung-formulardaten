from paddleocr import PaddleOCR, draw_ocr
import cv2
import os
import numpy as np

def detect_boxes(img_path, filename):
    """
    Erkennt Bounding-Boxen in einem Bild und gibt sie zurück.
    
    Args:
        img_path: Pfad zum Bild
        filename: Name für die Ausgabedatei
    
    Returns:
        Tuple mit (coords, boxes, texts, scores, centers, box_heights)
        centers enthält die Mittelpunkte der Boxen (x, y)
        box_heights enthält die Höhe jeder Box
    """
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='de')
    result = ocr.ocr(img_path, cls=True)

    box_coords, boxes, texts, scores, centers, box_heights = [], [], [], [], [], []

    for line in result:
        if (line):
            for box, (text, confidence) in line:
                # Box-Koordinaten extrahieren
                box = np.array(box)
                
                # Mittelpunkt der Box berechnen
                center_x = np.mean(box[:, 0])
                center_y = np.mean(box[:, 1])
                
                # Box-Höhe berechnen (maximale Y-Koordinate - minimale Y-Koordinate)
                box_height = np.max(box[:, 1]) - np.min(box[:, 1])

                # Get coordinates of the box corners
                box_coordinates = [
                    (int(box[0][0]), int(box[0][1])),  # Top-left
                    (int(box[1][0]), int(box[1][1])),  # Top-right
                    (int(box[2][0]), int(box[2][1])),  # Bottom-right
                    (int(box[3][0]), int(box[3][1]))   # Bottom-left
                ]

                box_coords.append(box_coordinates)
                boxes.append(box)
                texts.append(text)
                scores.append(confidence)
                centers.append((center_x, center_y))
                box_heights.append(box_height)
                
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ausgabeverzeichnis erstellen, falls es nicht existiert
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Annotationen auf dem Bild zeichnen
    annotated = draw_ocr(img, boxes, texts, scores, font_path='src/fonts/german.ttf')
    
    # Bild speichern
    cv2.imwrite('output/detected_hw/' + filename + ".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    
    return box_coords, boxes, texts, scores, centers, box_heights

def match_boxes(label_centers, handwritten_centers, label_texts, handwritten_texts, label_heights=None, handwritten_heights=None, y_threshold=30):
    """
    Ordnet Label-Boxen und handgeschriebene Boxen anhand ihrer Y-Koordinaten zu.
    Speichert für jede Zeile die Y-Koordinaten der größten erkannten Box.
    
    Args:
        label_centers: Liste der Mittelpunkte der Label-Boxen
        handwritten_centers: Liste der Mittelpunkte der handgeschriebenen Boxen
        label_texts: Liste der Label-Texte
        handwritten_texts: Liste der handgeschriebenen Texte
        label_heights: Liste der Höhen der Label-Boxen
        handwritten_heights: Liste der Höhen der handgeschriebenen Boxen
        y_threshold: Maximaler Y-Abstand für eine Zuordnung
    
    Returns:
        Tuple mit (matches, row_y_coordinates)
        matches: Liste von Tupeln (label_text, handwritten_text, y_diff)
        row_y_coordinates: Liste der Y-Koordinaten der größten Box in jeder Zeile
    """
    matches = []
    
    # Für jedes Label den besten passenden handgeschriebenen Text finden
    for i, (label_x, label_y) in enumerate(label_centers):
        best_match = None
        min_y_diff = float('inf')
        
        for j, (hw_x, hw_y) in enumerate(handwritten_centers):
            # Y-Differenz berechnen (absoluter Wert)
            y_diff = abs(label_y - hw_y)
            
            # Wenn die Y-Differenz innerhalb des Schwellenwerts liegt und kleiner als die bisherige beste Übereinstimmung ist
            if y_diff < y_threshold and y_diff < min_y_diff:
                min_y_diff = y_diff
                best_match = j
        
        # Wenn eine Übereinstimmung gefunden wurde
        if best_match is not None:
            matches.append((label_texts[i], handwritten_texts[best_match], min_y_diff))
    
    # Y-Koordinaten der größten Box in jeder Zeile speichern
    # Wir gruppieren die Matches nach ähnlichen Y-Koordinaten und berücksichtigen die Box-Höhe
    row_y_coordinates = []
    row_boxes = []
    
    if matches:
        # Erstelle eine Liste mit allen Label-Boxen und ihren Eigenschaften
        label_boxes = []
        for i, (label_x, label_y) in enumerate(label_centers):
            # Stelle sicher, dass die Box-Höhe korrekt verwendet wird
            box_height = label_heights[i] if label_heights and i < len(label_heights) else 0
            label_boxes.append({
                'index': i,
                'text': label_texts[i],
                'y': label_y,
                'height': box_height,
                'center_x': label_x
            })
        
        # Sortiere Boxen nach Y-Koordinate
        sorted_boxes = sorted(label_boxes, key=lambda x: x['y'])
        
        if sorted_boxes:
            # Initialisiere die erste Zeile
            current_row = [sorted_boxes[0]]
            # Berechne den mittleren Y-Wert der aktuellen Zeile
            current_row_y = sorted_boxes[0]['y']
            
            # Gruppiere Boxen in Zeilen basierend auf Y-Koordinaten und Überlappung
            for box in sorted_boxes[1:]:
                # Wenn die Box nahe genug an der aktuellen Zeile ist, füge sie zur Zeile hinzu
                if abs(box['y'] - current_row_y) < y_threshold:
                    current_row.append(box)
                    # Aktualisiere den mittleren Y-Wert der Zeile
                    current_row_y = sum(b['y'] for b in current_row) / len(current_row)
                else:
                    # Speichere die aktuelle Zeile und beginne eine neue
                    row_boxes.append(current_row)
                    current_row = [box]
                    current_row_y = box['y']
            
            # Füge die letzte Zeile hinzu
            if current_row:
                row_boxes.append(current_row)
            
            # Für jede Zeile, finde die größte Box und speichere ihre Y-Koordinate
            for row in row_boxes:
                if row:  # Stelle sicher, dass die Zeile nicht leer ist
                    # Finde die Box mit der größten Höhe in dieser Zeile
                    largest_box = max(row, key=lambda x: x['height'])
                    # Verwende die Y-Koordinate der größten Box für die Zeile
                    row_y_coordinates.append(largest_box['y'])
                    print(f"Zeile mit Texten: {[box['text'] for box in row]}, Y-Koordinate der größten Box: {largest_box['y']:.1f}px, Höhe: {largest_box['height']:.1f}px")
    
    print(f"Y-Koordinaten der Zeilen: {row_y_coordinates}")
    
    return matches, row_y_coordinates

def visualize_matches(label_img_path, handwritten_img_path, matches, row_y_coordinates=None, output_path='output/matched_boxes.png'):
    """
    Visualisiert die zugeordneten Boxen auf einem kombinierten Bild.
    
    Args:
        label_img_path: Pfad zum Label-Bild
        handwritten_img_path: Pfad zum Bild mit handgeschriebenem Text
        matches: Liste der zugeordneten Texte
        row_y_coordinates: Liste der Y-Koordinaten der größten Box in jeder Zeile
        output_path: Pfad für die Ausgabedatei
    """
    # Bilder laden
    label_img = cv2.imread(label_img_path)
    handwritten_img = cv2.imread(handwritten_img_path)
    initial_img = cv2.imread("images/cropped_verstorbenen_section.jpg")
    
    # Sicherstellen, dass beide Bilder die gleiche Höhe haben
    if label_img.shape[0] != handwritten_img.shape[0]:
        # Größeres Bild auf die Höhe des kleineren skalieren
        if label_img.shape[0] > handwritten_img.shape[0]:
            scale = handwritten_img.shape[0] / label_img.shape[0]
            label_img = cv2.resize(label_img, (int(label_img.shape[1] * scale), handwritten_img.shape[0]))
        else:
            scale = label_img.shape[0] / handwritten_img.shape[0]
            handwritten_img = cv2.resize(handwritten_img, (int(handwritten_img.shape[1] * scale), label_img.shape[0]))
    
    # Bilder nebeneinander anordnen
    combined_img = np.hstack((label_img, handwritten_img))
    
    # Übereinstimmungen als Text anzeigen
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    
    for i, (label_text, hw_text, y_diff) in enumerate(matches):
        text = f"{i+1}. {label_text} -> {hw_text} (diff: {y_diff:.1f}px)"
        cv2.putText(combined_img, text, (10, y_offset), font, 0.7, (0, 0, 255), 2)
        y_offset += 30
    
    # Horizontale Linien für die Zeilenkoordinaten zeichnen, falls vorhanden
    if row_y_coordinates:
        # Zeichne einen Rahmen um die Zeilenbereiche
        for i, y_coord in enumerate(row_y_coordinates):
            # Bestimme die Y-Grenzen für diese Zeile
            y_top = int(y_coord) - 20  # Obere Grenze (20px über der Zeilen-Y-Koordinate)
            
            # Untere Grenze: entweder 20px unter der aktuellen Zeile oder bis zur nächsten Zeile
            if i < len(row_y_coordinates) - 1:
                y_bottom = min(int(y_coord) + 20, int(row_y_coordinates[i+1]) - 5)
            else:
                y_bottom = int(y_coord) + 20
            
            # Zeichne horizontale Linien für die Zeile
            # Hauptlinie (grün)
            cv2.line(combined_img, (0, int(y_coord)), (combined_img.shape[1], int(y_coord)), (0, 255, 0), 2)
            
            # Obere und untere Begrenzungslinien (gestrichelt, gelb)
            for x in range(0, combined_img.shape[1], 20):  # Gestrichelte Linie
                cv2.line(combined_img, (x, y_top), (x + 10, y_top), (0, 255, 255), 1)  # Obere Linie
                cv2.line(combined_img, (x, y_bottom), (x + 10, y_bottom), (0, 255, 255), 1)  # Untere Linie
                cv2.imwrite("output/row/row_" + str(i+1) + ".png", handwritten_img[y_top:y_bottom, :])
            
            # Zeilennummer und Y-Koordinate anzeigen
            label_text = f"Zeile {i+1}: y={int(y_coord)}px"
            cv2.putText(combined_img, label_text, (combined_img.shape[1] - 250, int(y_coord) - 10), 
                        font, 0.7, (0, 255, 0), 2)
            
            # Zeichne eine gestrichelte Linie für bessere Sichtbarkeit der Hauptlinie
            for x in range(0, combined_img.shape[1], 20):  # Gestrichelte Linie
                cv2.line(combined_img, (x, int(y_coord) - 2), (x + 10, int(y_coord) - 2), (255, 255, 0), 1)
    
    # Bild speichern
    cv2.imwrite(output_path, combined_img)
    print(f"Kombiniertes Bild mit Zuordnungen gespeichert unter: {output_path}")
    if row_y_coordinates:
        print(f"Horizontale Linien für {len(row_y_coordinates)} Zeilen wurden eingezeichnet.")
        
    return combined_img

def crop_image_to_rows(img_path, row_y_coordinates, output_dir='output/rows'):
    """
    Schneidet ein Bild in Zeilen basierend auf den Y-Koordinaten der Zeilen.
    
    Args:
        img_path: Pfad zum Bild, das in Zeilen geschnitten werden soll
        row_y_coordinates: Liste der Y-Koordinaten der größten Box in jeder Zeile
        output_dir: Verzeichnis für die Ausgabedateien
    
    Returns:
        Liste der Pfade zu den gespeicherten Zeilenbildern
    """
    # Bild laden
    img = cv2.imread(img_path)
    
    # Ausgabeverzeichnis erstellen, falls es nicht existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    row_images = []
    
    # Für jede Zeile
    for i, y_coord in enumerate(row_y_coordinates):
        # Bestimme die Y-Grenzen für diese Zeile
        y_top = int(y_coord) - 20  # Obere Grenze (20px über der Zeilen-Y-Koordinate)
        
        # Untere Grenze: entweder 20px unter der aktuellen Zeile oder bis zur nächsten Zeile
        if i < len(row_y_coordinates) - 1:
            y_bottom = min(int(y_coord) + 20, int(row_y_coordinates[i+1]) - 5)
        else:
            y_bottom = int(y_coord) + 20
        
        # Sicherstellen, dass die Grenzen innerhalb des Bildes liegen
        y_top = max(0, y_top)
        y_bottom = min(img.shape[0], y_bottom)
        
        # Zeile ausschneiden
        row_img = img[y_top:y_bottom, :]
        
        # Dateiname generieren
        output_path = os.path.join(output_dir, f"row_{i+1}.png")
        
        # Bild speichern
        cv2.imwrite(output_path, row_img)
        row_images.append(output_path)
        
        print(f"Zeile {i+1} gespeichert unter: {output_path}")
    
    return row_images