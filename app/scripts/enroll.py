"""Interactive webcam tool for collecting body and optional face references."""

import time
import cv2
from ultralytics import YOLO

from app.config import Settings
from app.core.face import FaceProcessor


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def count_images(directory):
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def open_webcam():
    """Open the default Windows webcam, with a fallback backend."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def capture_face_images(settings, face_dir, max_faces=10):
    """Capture face references using one webcam window and reliable key handling."""
    face_processor = FaceProcessor(
        model_name=settings.face_model,
        det_size=settings.face_det_size,
    )

    count = count_images(face_dir)
    if count >= max_faces:
        print(f"Face gallery already has {count} images (limit: {max_faces}).")
        return

    cap = open_webcam()
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "Face Enrollment - SPACE saves, ESC exits"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nFACE ENROLLMENT")
    print("Click inside the webcam window.")
    print("SPACE: Save detected face | ESC: Exit\n")

    face_crop = None
    face_bbox = None
    last_detection_time = 0.0
    frame_number = 0

    try:
        while count < max_faces:
            ok, frame = cap.read()
            if not ok:
                print("Webcam frame could not be read.")
                break

            frame_number += 1
            if frame_number % 3 == 0:
                try:
                    faces = face_processor.extract_faces(frame)
                    if faces:
                        face = max(
                            faces,
                            key=lambda f: max(0, f.bbox[2] - f.bbox[0])
                            * max(0, f.bbox[3] - f.bbox[1]),
                        )
                        x1, y1, x2, y2 = map(int, face.bbox)
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            crop = frame[y1:y2, x1:x2]
                            if crop.size:
                                face_crop = cv2.resize(crop, (112, 112))
                                face_bbox = (x1, y1, x2, y2)
                                last_detection_time = time.time()
                    else:
                        face_crop = None
                        face_bbox = None
                        last_detection_time = 0.0
                except Exception as exc:
                    print(f"Face detection error: {exc}")
                    face_crop = None
                    face_bbox = None
                    last_detection_time = 0.0

            valid = (
                face_crop is not None
                and time.time() - last_detection_time <= 2.0
            )
            display = frame.copy()

            if valid:
                status = f"FACE DETECTED | SAVED: {count}/{max_faces}"
                if face_bbox:
                    x1, y1, x2, y2 = face_bbox
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                color = (0, 255, 0)
            else:
                status = f"Looking for face... | SAVED: {count}/{max_faces}"
                color = (0, 200, 255)

            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(display, "SPACE: SAVE    ESC: EXIT", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(window_name, display)

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Enrollment window closed.")
                    break
            except cv2.error:
                break

            key = cv2.waitKeyEx(10) & 0xFF
            if key == 27:
                print("ESC pressed. Closing face enrollment.")
                break
            if key == 32:
                if not valid:
                    print("Not saved: no face detected. Keep your face visible.")
                    continue
                path = face_dir / f"{count:04d}.jpg"
                if cv2.imwrite(str(path), face_crop.copy()):
                    count += 1
                    print(f"Saved face image: {path} ({count}/{max_faces})")
                    face_crop = None
                    face_bbox = None
                    last_detection_time = 0.0
                else:
                    print(f"ERROR: Could not write image: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print(f"Face enrollment finished. Total face images: {count}")


def capture_body_images(settings, body_dir, max_images=10):
    """Capture person/body crops in a single OpenCV window."""
    model_path = settings.project_root / "Models" / "yolo11s-seg.pt"
    if not model_path.exists():
        print(f"Error: Segmentation model not found: {model_path}")
        return

    print(f"Loading segmentation model: {model_path}")
    model = YOLO(str(model_path))

    count = count_images(body_dir)
    if count >= max_images:
        print(f"Body gallery already has {count} images (limit: {max_images}).")
        return

    cap = open_webcam()
    if not cap.isOpened():
        print("Error: Could not open webcam. Close other camera applications.")
        return

    window_name = "Body Enrollment - SPACE saves, ESC exits"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\nBODY ENROLLMENT")
    print("Click inside the Body Enrollment webcam window.")
    print("SPACE: Save detected person | ESC: Exit")
    print("Keep your upper or full body visible and stand in good lighting.\n")

    body_crop = None
    body_ready = False

    try:
        while count < max_images:
            ok, frame = cap.read()
            if not ok:
                print("Webcam frame could not be read.")
                break

            display = frame.copy()
            body_crop = None
            body_ready = False
            status = "Looking for a person..."
            status_color = (0, 200, 255)

            try:
                result = model.predict(
                    frame,
                    conf=0.40,
                    classes=[0],
                    imgsz=640,
                    verbose=False,
                )[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    # Use the largest detected person in the frame.
                    best_idx = max(
                        range(len(result.boxes)),
                        key=lambda i: (
                            float(result.boxes[i].xyxy[0][2] - result.boxes[i].xyxy[0][0])
                            * float(result.boxes[i].xyxy[0][3] - result.boxes[i].xyxy[0][1])
                        ),
                    )
                    box = result.boxes[best_idx]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])

                    h, w = frame.shape[:2]
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        # A rectangular person crop is more reliable for enrollment
                        # than resizing a segmentation mask that may be misaligned.
                        person_roi = frame[y1:y2, x1:x2]
                        if person_roi.size:
                            body_crop = cv2.resize(person_roi, (128, 256))
                            body_ready = True
                            status = f"PERSON DETECTED | READY | SAVED: {count}/{max_images}"
                            status_color = (0, 255, 0)

                            cv2.rectangle(display, (x1, y1), (x2, y2),
                                          (0, 255, 0), 2)
                            cv2.putText(display, f"Person {confidence:.2f}",
                                        (x1, max(25, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                        (0, 255, 0), 2)
                else:
                    status = f"No person detected | SAVED: {count}/{max_images}"

            except Exception as exc:
                status = f"Detection error: {str(exc)[:65]}"
                body_crop = None
                body_ready = False

            cv2.putText(display, status[:100], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        status_color, 2)
            cv2.putText(display, "SPACE: SAVE    ESC: EXIT", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)
            cv2.imshow(window_name, display)

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Enrollment window closed.")
                    break
            except cv2.error:
                break

            # Only this one window receives keyboard input; no crop-preview
            # windows are opened to steal focus.
            key = cv2.waitKeyEx(10) & 0xFF

            if key == 27:
                print("ESC pressed. Closing body enrollment.")
                break

            if key == 32:
                if not body_ready or body_crop is None:
                    print("Not saved: no person crop is ready. Wait for the green box.")
                    continue

                body_path = body_dir / f"{count:04d}.jpg"
                if cv2.imwrite(str(body_path), body_crop.copy()):
                    count += 1
                    print(f"Saved body image: {body_path} ({count}/{max_images})")
                    body_crop = None
                    body_ready = False
                    if count >= max_images:
                        print("Maximum number of body images reached.")
                        break
                else:
                    print(f"ERROR: Could not write body image: {body_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    print(f"Body enrollment finished. Total body images: {count}")


def main():
    """Capture person reference images from a webcam."""
    settings = Settings()
    person_name = input("Person Name: ").strip()

    if not person_name:
        print("Error: Person name cannot be empty.")
        return

    body_dir = settings.gallery_dir / person_name / "images"
    face_dir = settings.gallery_dir / person_name / "faces"
    body_dir.mkdir(parents=True, exist_ok=True)
    face_dir.mkdir(parents=True, exist_ok=True)

    mode = input("Enrollment mode (body/face): ").strip().lower()
    if mode == "face":
        capture_face_images(settings, face_dir, max_faces=10)
    elif mode == "body":
        capture_body_images(settings, body_dir, max_images=10)
    else:
        print("Invalid mode. Enter 'body' or 'face'.")


if __name__ == "__main__":
    main()
