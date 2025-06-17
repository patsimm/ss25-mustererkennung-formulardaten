# Automatisiertes Auslesen von handschriftlichen Texten aus Formularen

Dieses Projekt implementiert ein System zur automatisierten Erkennung und Extraktion von handschriftlichen Texten aus Formularen. Es kombiniert Computer Vision Techniken mit Deep Learning für die Texterkennung.

Die komplette Implementierungslogik ist im Jupyter Notebook `layout.ipynb` zu finden, welches den gesamten Workflow von der Bildvorverarbeitung bis zur Textextraktion dokumentiert.

## Konzept

Das System besteht aus mehreren Hauptkomponenten:

1. **Dokumenten-Scanning und Vorverarbeitung**
   - Automatische Erkennung und Perspektivkorrektur von Formularen
   - Bildvorverarbeitung zur Optimierung der Texterkennung
   - Entfernung von Hintergrundrauschen und Verbesserung der Bildqualität

2. **Layout-Analyse**
   - Erkennung von Formularfeldern und deren Position
   - Segmentierung der einzelnen Eingabefelder
   - Identifikation von relevanten Bereichen für die Textextraktion
   - Verwendung von PaddleOCR zur zuverlässigen Erkennung von Handschrift-Bounding Boxes

3. **Handschriftenerkennung**
   - Verwendung des TrOCR (Transformer OCR) Modells
   - Fine-tuning auf spezifische Handschriftendaten
   - Extraktion des Textes aus den erkannten Formularfeldern

## Technische Implementierung

Das Projekt nutzt folgende Haupttechnologien:

- **OpenCV** für Bildverarbeitung und Dokumentenscanning
- **PyTorch** und **Transformers** für das Deep Learning Modell
- **TrOCR** als Basis für die Handschriftenerkennung
- **PaddleOCR** für die präzise Erkennung von Handschrift-Bounding Boxes
- **Python** als Hauptprogrammiersprache

## Verwendung

1. **Vorbereitung der Daten**
   - Scannen der Formulare
   - Sicherstellen einer ausreichenden Bildqualität
   - Handschriftliche Texte müssen in blauer Tinte geschrieben sein

2. **Verarbeitung**
   - Automatische Erkennung und Korrektur der Formularperspektive
   - Extraktion der einzelnen Formularfelder
   - Erkennung der handschriftlichen Texte
   - Zuordnung zu den entsprechenden Formularfeldern

3. **Ausgabe**
   - Strukturierte Ausgabe der erkannten Texte

## Anforderungen

- Python 3.8+
- CUDA-fähige GPU für optimale Performance
- Ausreichend RAM für die Bildverarbeitung